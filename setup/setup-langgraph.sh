#!/usr/bin/env bash
# Install and start: Gemma 4 (llama-server), LangGraph FastAPI, Next.js chat UI.
# Run: sudo ./setup/setup-langgraph.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GEMMA4_DIR="${REPO_DIR}/gemma4"
CACHE_DIR="/data/huggingface"
IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-thor"
GEMMA_SERVICE="gemma4.service"
GEMMA_SERVICE_PATH="/etc/systemd/system/${GEMMA_SERVICE}"
GEMMA_PORT="8080"
LANGGRAPH_SERVICE="langgraph.service"
LANGGRAPH_SERVICE_PATH="/etc/systemd/system/${LANGGRAPH_SERVICE}"
LANGGRAPH_PORT="8008"
NEXT_PORT="3000"
VENV_PATH="${REPO_DIR}/langgraph-venv"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-$((4 * 60 * 60))}"
WAIT_POLL_SECONDS="10"
require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Please run as root: sudo $0" >&2
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "Docker is not installed and this installer only knows apt-get for Docker." >&2
      exit 1
    fi
    echo "Installing docker.io…"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y docker.io docker-compose-v2
  fi

  systemctl enable --now containerd docker.socket docker
  systemctl restart containerd docker

  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart containerd docker
  fi
}

ensure_python_venv_basics() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y python3 python3-venv python3-pip curl
}

ensure_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y nodejs npm
}

ensure_oak_udev() {
  # Luxonis OAK / Movidius USB devices need a udev rule so non-root users can open them.
  # Without this, depthai (and depthai-viewer) fail to claim the device after firmware boot.
  local rules_path="/etc/udev/rules.d/80-movidius.rules"
  local rule='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
  if [[ ! -f "${rules_path}" ]] || ! grep -qF "${rule}" "${rules_path}"; then
    echo "Installing OAK/Movidius udev rule at ${rules_path}…"
    echo "${rule}" > "${rules_path}"
    chmod 0644 "${rules_path}"
    udevadm control --reload-rules
    udevadm trigger
    echo "Note: replug the OAK camera so it re-enumerates under the new rule."
  fi
}

gemma_ready() {
  curl -fsS "http://127.0.0.1:${GEMMA_PORT}/health" >/dev/null 2>&1 || \
    curl -fsS "http://127.0.0.1:${GEMMA_PORT}/" >/dev/null 2>&1
}

langgraph_api_ready() {
  curl -fsS "http://127.0.0.1:${LANGGRAPH_PORT}/health" >/dev/null 2>&1
}

install_gemma_systemd() {
  require_file "${GEMMA4_DIR}/gemma4-run.sh"
  require_file "${GEMMA4_DIR}/gemma4.service"
  install -d -m 0755 "$(dirname "${GEMMA_SERVICE_PATH}")"
  sed "s|@REPO_DIR@|${REPO_DIR}|g" "${GEMMA4_DIR}/gemma4.service" > "${GEMMA_SERVICE_PATH}"
  chmod 0644 "${GEMMA_SERVICE_PATH}"
}

install_langgraph_systemd() {
  require_file "${REPO_DIR}/langgraph-service/langgraph.service"
  sed "s|@REPO_DIR@|${REPO_DIR}|g" "${REPO_DIR}/langgraph-service/langgraph.service" > "${LANGGRAPH_SERVICE_PATH}"
  chmod 0644 "${LANGGRAPH_SERVICE_PATH}"
}

wait_for_gemma() {
  local now started elapsed
  started="$(date +%s)"
  echo "Waiting for Gemma (llama-server) on port ${GEMMA_PORT}…"
  while true; do
    if gemma_ready; then
      echo "Gemma is ready."
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - started))
    if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out after ${WAIT_TIMEOUT_SECONDS}s waiting for Gemma." >&2
      echo "Check: journalctl -u ${GEMMA_SERVICE} -n 50 --no-pager" >&2
      exit 1
    fi
    printf '[%4ds] waiting for http://127.0.0.1:%s/ …\n' "${elapsed}" "${GEMMA_PORT}"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

wait_for_langgraph() {
  local s=0
  echo "Waiting for LangGraph API on port ${LANGGRAPH_PORT}…"
  while ! langgraph_api_ready; do
    s=$((s + 1))
    if (( s > 60 )); then
      echo "LangGraph /health not responding. Check: journalctl -u ${LANGGRAPH_SERVICE} -n 40" >&2
      exit 1
    fi
    sleep 1
  done
  echo "LangGraph API is ready."
}

install_langgraph_venv() {
  if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "Creating ${VENV_PATH}…"
    python3 -m venv "${VENV_PATH}"
  fi
  "${VENV_PATH}/bin/pip" install -U pip setuptools wheel
  "${VENV_PATH}/bin/pip" install -r "${REPO_DIR}/langgraph-service/requirements.txt"
}

write_next_env() {
  local p="${REPO_DIR}/web/.env.local"
  echo "NEXT_PUBLIC_LANGGRAPH_API_URL=http://127.0.0.1:${LANGGRAPH_PORT}" > "${p}"
  echo "Wrote ${p}"
}

ensure_run_dirs_owned_by_invoker() {
  # `install -d` as root (via `sudo ./setup-*.sh`) creates root-owned `run/`, which breaks
  # dev services started as a normal user (e.g. uv writing temp uploads to `run/molmo-uploads/`).
  local u="${EUID:-$(id -u)}"
  if [[ "${u}" -eq 0 && -n "${SUDO_USER:-}" ]]; then
    # Prefer numeric uid/gid for the invoking user (avoids "group not found" edge cases)
    local su_root_uid su_root_gid
    su_root_uid="${SUDO_UID:-$(id -u "${SUDO_USER}")}"
    su_root_gid="$(id -g "${SUDO_USER}")"
    chown -R "${su_root_uid}:${su_root_gid}" \
      "${REPO_DIR}/run" "${REPO_DIR}/web" 2>/dev/null || true
  fi
}

stop_next_if_pidfile() {
  local pf="${REPO_DIR}/run/next.pid"
  if [[ -f "${pf}" ]]; then
    local pid
    pid="$(tr -d '\r\n' < "${pf}" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping previous Next.js (pid ${pid})…"
      kill "${pid}" 2>/dev/null || true
      sleep 1
    fi
    rm -f "${pf}"
  fi
}

start_next_dev() {
  require_cmd node
  require_cmd npm
  install -d -m 0755 "${REPO_DIR}/run" "${REPO_DIR}/run/molmo-uploads" "${REPO_DIR}/web"
  ensure_run_dirs_owned_by_invoker
  stop_next_if_pidfile
  (cd "${REPO_DIR}/web" && npm install)
  write_next_env
  echo "Starting Next.js on 0.0.0.0:${NEXT_PORT} (background)…"
  cd "${REPO_DIR}/web"
  nohup ./node_modules/.bin/next dev -H 0.0.0.0 -p "${NEXT_PORT}" \
    >> "${REPO_DIR}/run/next.log" 2>&1 &
  echo $! > "${REPO_DIR}/run/next.pid"
  sleep 2
  if [[ -f "${REPO_DIR}/run/next.pid" ]]; then
    echo "Next.js pid $(cat "${REPO_DIR}/run/next.pid") — log: ${REPO_DIR}/run/next.log"
  fi
}

print_urls() {
  echo
  echo "=== LangGraph + Gemma stack ==="
  echo "  Gemma (llama-server) : http://127.0.0.1:${GEMMA_PORT}  (internal to this host)"
  echo "  LangGraph API        : http://127.0.0.1:${LANGGRAPH_PORT}"
  echo "  Next.js chat         : http://127.0.0.1:${NEXT_PORT}  (and LAN http://<this-host>:${NEXT_PORT})"
  echo
  echo "  systemctl status ${GEMMA_SERVICE} --no-pager -l"
  echo "  systemctl status ${LANGGRAPH_SERVICE} --no-pager -l"
  echo "  tail -f ${REPO_DIR}/run/next.log"
  echo
}

main() {
  require_root
  require_cmd systemctl
  require_cmd sed
  ensure_docker
  ensure_python_venv_basics
  ensure_node
  ensure_oak_udev
  require_cmd docker
  require_cmd curl

  require_file "${REPO_DIR}/langgraph-service/app.py"
  require_file "${REPO_DIR}/langgraph-service/state_graph.py"
  require_file "${REPO_DIR}/langgraph-service/requirements.txt"

  echo "Cache directory: ${CACHE_DIR}"
  install -d -m 0755 "${CACHE_DIR}" "${CACHE_DIR}/hub" "${CACHE_DIR}/xet"

  echo "Pulling Gemma container image (if not present)…"
  docker pull "${IMAGE}"

  install_gemma_systemd
  install_langgraph_venv
  install_langgraph_systemd

  systemctl daemon-reload
  systemctl enable "${GEMMA_SERVICE}" "${LANGGRAPH_SERVICE}"
  systemctl restart "${GEMMA_SERVICE}"

  wait_for_gemma
  systemctl restart "${LANGGRAPH_SERVICE}"
  wait_for_langgraph

  start_next_dev
  print_urls
}

main "$@"
