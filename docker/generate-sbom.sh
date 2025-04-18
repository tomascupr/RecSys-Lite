#!/bin/bash
# Script to generate a Software Bill of Materials (SBOM) for RecSys-Lite using Syft

set -e

# Check if Syft is installed
if ! command -v syft &> /dev/null; then
    echo "Syft is required but not installed. Installing..."
    # Install Syft
    curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
fi

# Directory for SBOM output
mkdir -p sbom

# Generate SBOM for each Docker image
echo "Generating SBOM for recsys-lite-api..."
syft recsys-lite-api:latest -o json > sbom/recsys-lite-api.sbom.json
syft recsys-lite-api:latest -o spdx-json > sbom/recsys-lite-api.spdx.json
syft recsys-lite-api:latest -o cyclonedx-json > sbom/recsys-lite-api.cyclonedx.json

echo "Generating SBOM for recsys-lite-worker..."
syft recsys-lite-worker:latest -o json > sbom/recsys-lite-worker.sbom.json
syft recsys-lite-worker:latest -o spdx-json > sbom/recsys-lite-worker.spdx.json
syft recsys-lite-worker:latest -o cyclonedx-json > sbom/recsys-lite-worker.cyclonedx.json

# Generate combined SBOM for Python project
echo "Generating SBOM for Python project..."
syft dir:.. -o json > sbom/recsys-lite-project.sbom.json
syft dir:.. -o spdx-json > sbom/recsys-lite-project.spdx.json
syft dir:.. -o cyclonedx-json > sbom/recsys-lite-project.cyclonedx.json

echo "SBOM generation complete. Files saved to sbom/ directory:"
ls -la sbom/

echo "Summary of dependencies:"
jq '.artifacts | length' sbom/recsys-lite-project.sbom.json
echo "Top-level Python dependencies:"
jq '.artifacts[] | select(.type=="python-package") | .name' sbom/recsys-lite-project.sbom.json | sort | uniq | head -20

echo "Done!"