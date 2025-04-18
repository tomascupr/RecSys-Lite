.PHONY: build run run-prod test lint clean sbom all help

# Set environment variables
DOCKER_COMPOSE = docker compose
BUILD_ENV ?= development
VERSION ?= latest

help:
	@echo "RecSys-Lite Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make build         Build Docker images"
	@echo "  make run           Run in development mode"
	@echo "  make run-prod      Run in production mode"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linter"
	@echo "  make clean         Clean up containers and images"
	@echo "  make sbom          Generate Software Bill of Materials (SBOM)"
	@echo "  make all           Build, test, and run in development mode"
	@echo ""
	@echo "Environment variables:"
	@echo "  BUILD_ENV=development|production (Default: development)"
	@echo "  VERSION=x.y.z (Default: latest)"

build:
	@echo "Building RecSys-Lite Docker images..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml build

run:
	@echo "Running RecSys-Lite in development mode..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml up -d

run-prod:
	@echo "Running RecSys-Lite in production mode..."
	BUILD_ENV=production VERSION=$(VERSION) $(DOCKER_COMPOSE) -f docker/docker-compose.prod.yml up -d

test:
	@echo "Running tests..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.test.yml up --abort-on-container-exit test

lint:
	@echo "Running linters..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.test.yml up --abort-on-container-exit lint

check-image-size:
	@echo "Checking Docker image size..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.test.yml up --abort-on-container-exit image-size-check

sbom:
	@echo "Generating Software Bill of Materials (SBOM)..."
	@mkdir -p docker/sbom
	@bash docker/generate-sbom.sh

clean:
	@echo "Cleaning up containers and images..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down --remove-orphans
	$(DOCKER_COMPOSE) -f docker/docker-compose.prod.yml down --volumes --remove-orphans
	$(DOCKER_COMPOSE) -f docker/docker-compose.test.yml down --remove-orphans
	@echo "Removing images..."
	-docker rmi recsys-lite-api:$(VERSION) recsys-lite-worker:$(VERSION)

all: build test run