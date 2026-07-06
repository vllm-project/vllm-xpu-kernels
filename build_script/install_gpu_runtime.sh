#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Install GPU runtime .deb packages based on a profile from gpu_runtime_packages.json.
#
# Usage (inside Dockerfile):
#   COPY gpu_runtime_packages.json install_gpu_runtime.sh /tmp/
#   RUN bash /tmp/install_gpu_runtime.sh [PROFILE_NAME]
#
# If PROFILE_NAME is omitted or "default", the default profile from the JSON is used.

set -euo pipefail

CONFIG_FILE="${1:-/tmp/gpu_runtime_packages.json}"
PROFILE="${2:-default}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Resolve "default" to the actual profile name
if [ "${PROFILE}" = "default" ]; then
    PROFILE=$(python3 -c "
import json, sys
cfg = json.load(open('${CONFIG_FILE}'))
print(cfg['default'])
")
fi

echo "==> Installing GPU runtime packages from profile: ${PROFILE}"

# Extract the list of URLs for the selected profile
URLS=$(python3 -c "
import json, sys
cfg = json.load(open('${CONFIG_FILE}'))
profile = cfg['profiles'].get('${PROFILE}')
if profile is None:
    print(f'ERROR: Profile \"${PROFILE}\" not found. Available: {list(cfg[\"profiles\"].keys())}', file=sys.stderr)
    sys.exit(1)
print('\n'.join(profile['packages']))
")

if [ -z "${URLS}" ]; then
    echo "ERROR: No packages found for profile '${PROFILE}'"
    exit 1
fi

# Download and install
WORKDIR=$(mktemp -d)
echo "==> Downloading packages to ${WORKDIR} ..."

while IFS= read -r url; do
    echo "    ${url}"
    wget -q -P "${WORKDIR}" "${url}"
done <<< "${URLS}"

echo "==> Installing .deb packages ..."
ls -al "${WORKDIR}"
dpkg -i "${WORKDIR}"/intel-igc-core*
dpkg -i "${WORKDIR}"/*.deb #|| true
# apt-get install -f -y --no-install-recommends

echo "==> Cleaning up ..."
rm -rf "${WORKDIR}"

echo "==> GPU runtime installation complete (profile: ${PROFILE})"