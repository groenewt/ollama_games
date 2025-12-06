#!/bin/bash
# Compose configuration
export PROJECT_NAME="gametheory"
export PROJECT_DIR="/workspace"
export PROJECT_SCRIPT_DIR="/projscripts"
export PROJECT_UTILS="${PROJECT_SCRIPT_DIR}/utility"
export UV_LINK_MODE=copy
source "${PROJECT_UTILS}/common_utils.sh"

log_info "Running Project (${PROJECT_NAME} in ${PROJECT_DIR}) - scripts dir: ${PROJECT_SCRIPT_DIR}"


log_info "Sourcing Ollama config (ollama.env in ${PROJECT_DIR})- $( cat $PROJECT_DIR/ollama.env)"
{
  # Start Ollama service in the background
  source "${PROJECT_DIR}/ollama.env"
  log_success "Ollama Env loaded!"
} || {
  log_info "Failed to source ollama config (using default)..."
}

log_info "Launching Ollama Service"
  {
    banner_llama
    sleep 1
  } || {
      log_debug "Failed to display ollama banner"
}

{
  # Start Ollama service in the background
  /usr/bin/ollama serve &
  log_success "Launched Ollama Service"
  # Wait for Ollama to be ready
  sleep 3

} || {
  log_info "Failed to launch ollama..."
  exit 1
}

log_info "Checking model list"
{
  log_info "SOURCING: ${PROJECT_SCRIPT_DIR}/init/service_init.sh"
  source "${PROJECT_SCRIPT_DIR}/init/service_init.sh"
  log_success "Model List Checked/Pulled"

  {
    MODEL_LIST="${MODEL_LIST:-/tmp/model_list.txt}"
    if [ -f "${MODEL_LIST}" ]; then
      log_info "=== Removing Model List @ ${MODEL_LIST} ==="
    fi
  } || {
    log_debug "Error removing Model List"
  }

} || {   # <-- this now closes the OUTER block correctly
  log_warn "Failed to validate model list"
}
# Pull models from model_list.txt


log_info "Launching pyapp"
  {
    banner_python
    sleep 3
  } || {
      log_debug "Failed to display python banner"
}

log_info "Py App Check"
log_info "Py App UV management"

uv sync || log_fail "UV FAILURE!!!"

log_info "🚀 Starting Marimo on port ${MARIMO_PORT}..."
# Check if there are notebooks in the directory
PYAPP_PORT="${PYAPP_PORT:-2718}"
echo -e "APP RUNNING on:  localhost:${PYAPP_PORT}"

uv run marimo run "apps/arena.py"  --host 0.0.0.0
