#!/bin/bash

export PROJECT_UTILS="${PROJECT_SCRIPT_DIR}/utility"

source "${PROJECT_UTILS}/common_utils.sh"

OLLAMA_BIN="${OLLAMA_BIN:-/usr/bin/ollama}"
MODEL_LIST="${MODEL_LIST:-/tmp/model_list.txt}"
# Pull models from model_list.txt
log_info "=== Checking for Models ==="
if [ -f "${MODEL_LIST}" ]; then
    log_success "📋 Found model list at ${MODEL_LIST}"
    (
      while IFS= read -r model || [ -n "$model" ]; do
        model=$(echo "$model" | xargs)
        if [[ -n "$model" && ! "$model" =~ ^# && ! "$model" =~ ^EMBEDDING ]]; then
          log_info "   Found model: $model"
          log_info "⬇️  Pulling model: $model"
          ${OLLAMA_BIN} pull "$model" 2>&1 | sed 's/^/   /' # & <<
          {
              banner_llama_simple
              sleep 1
          } || {
              log_debug "Failed to display llama simple banner"
          }

        fi
      done < "${MODEL_LIST}"
    )
else
    log_warn "⚠️  No model list found at ${MODEL_LIST}"
fi
