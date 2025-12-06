#!/bin/env bash



# Project paths

export UTILITIES_DIR="${PROJECT_SCRIPT_DIR}/utility"
source "${UTILITIES_DIR}/misc/misc_utils_banner.sh"

# Color output
export COLOR_RED='\033[0;31m'
export COLOR_GREEN='\033[0;32m'
export COLOR_YELLOW='\033[1;33m'
export COLOR_BLUE='\033[0;34m'
export COLOR_MAGENTA='\033[0;35m'
export COLOR_CYAN='\033[0;36m'
export COLOR_NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${COLOR_MAGENTA} LOG - ${PROJECT_NAME} ${COLOR_NC} -${COLOR_GREEN}[INFO]${COLOR_NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${COLOR_MAGENTA} LOG - ${PROJECT_NAME} ${COLOR_NC} -${COLOR_YELLOW}[WARN]${COLOR_NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${COLOR_MAGENTA} LOG - ${PROJECT_NAME} ${COLOR_NC} -${COLOR_RED}[ERROR]${COLOR_NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_debug() {
    if [[ "${DEBUG:-0}" == "1" ]]; then
        echo -e "${COLOR_MAGENTA} LOG - ${PROJECT_NAME} ${COLOR_NC} -${COLOR_BLUE}[DEBUG]${COLOR_NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

log_success() {
    echo -e "${COLOR_MAGENTA} LOG - ${PROJECT_NAME} ${COLOR_NC} -${COLOR_GREEN}✓${COLOR_NC} $1"
}

log_fail() {
    echo -e "${COLOR_MAGENTA} LOG - ${PROJECT_NAME} ${COLOR_NC} -${COLOR_RED}✗${COLOR_NC} $1"
}