.PHONY: help up down purge pull-models pull restart-farms restart-farm1 restart-farm2 cp-data


export PROJECT_ROOT := $(shell pwd)




# =================================================================================
# UTILS
# =================================================================================
# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[0;32m
COLOR_BLUE := \033[0;34m
COLOR_YELLOW := \033[0;33m
COLOR_RED := \033[0;31m



# =================================================================================
# Make Targets
# =================================================================================

######
## HELP
######
help:
	@echo "$(COLOR_BOLD)Available targets:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)up$(COLOR_RESET)             - Start all containers"
	@echo "  $(COLOR_GREEN)down$(COLOR_RESET)           - Stop all containers"
	@echo "  $(COLOR_RED)purge$(COLOR_RESET)          - Stop containers, remove volumes and images"
	@echo "  $(COLOR_GREEN)pull-models$(COLOR_RESET)    - Pull all models from example_models.txt"
	@echo "  $(COLOR_GREEN)pull$(COLOR_RESET)           - Pull a single model (usage: make pull MODEL=qwen3:0.6b)"
	@echo "  $(COLOR_GREEN)restart-farms$(COLOR_RESET)  - Restart both llamafarm containers"
	@echo "  $(COLOR_GREEN)restart-farm1$(COLOR_RESET)  - Restart llamafarm1"
	@echo "  $(COLOR_GREEN)restart-farm2$(COLOR_RESET)  - Restart llamafarm2"
	@echo "  $(COLOR_GREEN)cp-data$(COLOR_RESET)        - Copy data volume to ./data"


######
## UP
######
up:
	@echo "$(COLOR_BLUE)Starting containers...$(COLOR_RESET)"
	docker compose up -d
	@echo "$(COLOR_GREEN)Containers started$(COLOR_RESET)"
######
## DOWN
######
down:
	@echo "$(COLOR_YELLOW)Stopping containers...$(COLOR_RESET)"
	docker compose down
	@echo "$(COLOR_GREEN)Containers stopped$(COLOR_RESET)"

######
## PURGE
######
purge:
	@echo "$(COLOR_RED)Purging containers, volumes, and images...$(COLOR_RESET)"
	docker compose down -v --rmi all
	@echo "$(COLOR_GREEN)Purge complete$(COLOR_RESET)"

######
## PULL MODELS
######
pull-models:
	@echo "$(COLOR_BLUE)Pulling models from example_models.txt...$(COLOR_RESET)"
	@failed=0; \
	while IFS= read -r model || [ -n "$$model" ]; do \
		model=$$(echo "$$model" | xargs); \
		case "$$model" in \
			""|\#*) continue ;; \
		esac; \
		echo "$(COLOR_YELLOW)Pulling: $$model$(COLOR_RESET)"; \
		if ! docker exec pyapp ollama pull "$$model"; then \
			echo "$(COLOR_RED)Failed to pull: $$model$(COLOR_RESET)"; \
			failed=1; \
		fi; \
	done < example_models.txt; \
	if [ $$failed -eq 1 ]; then \
		echo "$(COLOR_RED)Some models failed to pull$(COLOR_RESET)"; \
		exit 1; \
	fi; \
	echo "$(COLOR_GREEN)All models pulled$(COLOR_RESET)"

pull:
ifndef MODEL
	$(error MODEL is required. Usage: make pull MODEL=qwen3:0.6b)
endif
	@echo "$(COLOR_BLUE)Pulling model: $(MODEL)$(COLOR_RESET)"
	@if ! docker exec pyapp ollama pull $(MODEL); then \
		echo "$(COLOR_RED)Failed to pull model: $(MODEL)$(COLOR_RESET)"; \
		exit 1; \
	fi
	@echo "$(COLOR_GREEN)Model $(MODEL) pulled$(COLOR_RESET)"

######
## RESTART FARMS
######
restart-farms:
	@echo "$(COLOR_BLUE)Restarting llamafarms...$(COLOR_RESET)"
	docker compose restart llamafarm1 llamafarm2
	@echo "$(COLOR_GREEN)Llamafarms restarted$(COLOR_RESET)"

restart-farm1:
	@echo "$(COLOR_BLUE)Restarting llamafarm1...$(COLOR_RESET)"
	docker compose restart llamafarm1
	@echo "$(COLOR_GREEN)llamafarm1 restarted$(COLOR_RESET)"

restart-farm2:
	@echo "$(COLOR_BLUE)Restarting llamafarm2...$(COLOR_RESET)"
	docker compose restart llamafarm2
	@echo "$(COLOR_GREEN)llamafarm2 restarted$(COLOR_RESET)"

######
## DATA
######
cp-data:
	@echo "$(COLOR_BLUE)Copying data volume to ./data...$(COLOR_RESET)"
	@mkdir -p ./data
	docker cp pyapp:/data/. ./data/
	@echo "$(COLOR_GREEN)Data copied to ./data$(COLOR_RESET)"
