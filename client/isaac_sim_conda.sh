#!/usr/bin/env bash

# Isaac Sim with Conda Environment Wrapper
# This script runs Isaac Sim while using the conda environment for Python

set -e

# Configuration
CONDA_ENV_NAME="simgen"
ISAACSIM_PATH="/home/hongchix/main/isaacsim"
ISAACLAB_PATH="/home/hongchix/main/IsaacLab"

echo "[INFO] Starting Isaac Sim with conda environment '${CONDA_ENV_NAME}'..."
which python

# Set up Isaac Sim environment variables
export ISAACSIM_PATH="${ISAACSIM_PATH}"
export ISAACLAB_PATH="${ISAACLAB_PATH}"
export RESOURCE_NAME="IsaacSim"

# Isaac Sim Kit environment variables
export CARB_APP_PATH="${ISAACSIM_PATH}/kit"
export EXP_PATH="${ISAACSIM_PATH}/apps"
export ISAAC_PATH="${ISAACSIM_PATH}"

# Set up Python paths for Isaac Sim packages
ISAAC_PYTHON_PATHS="${ISAACSIM_PATH}/kit/python/lib/python3.10/site-packages"
ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/python_packages"
ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/exts/omni.isaac.kit"
ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/kit/kernel/py"
ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/kit/plugins/bindings-python"
ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/exts/omni.isaac.lula/pip_prebundle"
ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${ISAACSIM_PATH}/exts/omni.exporter.urdf/pip_prebundle"

# Check for additional pip_prebundle directories and add them
for pip_prebundle_dir in "${ISAACSIM_PATH}"/extscache/*/pip_prebundle; do
    if [ -d "${pip_prebundle_dir}" ]; then
        ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${pip_prebundle_dir}"
    fi
done

for pip_prebundle_dir in "${ISAACSIM_PATH}"/exts/*/pip_prebundle; do
    if [ -d "${pip_prebundle_dir}" ]; then
        ISAAC_PYTHON_PATHS="${ISAAC_PYTHON_PATHS}:${pip_prebundle_dir}"
    fi
done

export PYTHONPATH="${ISAAC_PYTHON_PATHS}:${PYTHONPATH}"

# Set up library paths for Isaac Sim
ISAAC_LIB_PATHS="${ISAACSIM_PATH}"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/kernel/plugins"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/libs/iray"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/bindings-python"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/carb_gfx"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/rtx"
ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${ISAACSIM_PATH}/kit/plugins/gpu.foundation"

# Add Isaac Sim library paths for schema plugins
for schema_lib in "${ISAACSIM_PATH}"/exts/omni.usd.schema.isaac/plugins/*/lib; do
    if [ -d "${schema_lib}" ]; then
        ISAAC_LIB_PATHS="${ISAAC_LIB_PATHS}:${schema_lib}"
    fi
done

export LD_LIBRARY_PATH="${ISAAC_LIB_PATHS}:${LD_LIBRARY_PATH}"

# Override Python executable to use conda python
export PYTHONEXE="$(which python)"

echo "[INFO] Using Python from conda environment: ${PYTHONEXE}"
echo "[INFO] Isaac Sim path: ${ISAACSIM_PATH}"
echo "[INFO] Starting Isaac Sim..."

# Change to Isaac Sim directory (some relative paths might be expected)
cd "${ISAACSIM_PATH}"

# Run Isaac Sim with Isaac Lab extensions if available
if [ -d "${ISAACLAB_PATH}/source/extensions" ]; then
    echo "[INFO] Including Isaac Lab extensions from: ${ISAACLAB_PATH}/source/extensions"
    exec "${ISAACSIM_PATH}/kit/kit" "${ISAACSIM_PATH}/apps/omni.isaac.sim.kit" \
        --ext-folder "${ISAACSIM_PATH}/apps" \
        --ext-folder "${ISAACLAB_PATH}/source/extensions" \
        "$@"
else
    echo "[INFO] Running Isaac Sim without Isaac Lab extensions"
    exec "${ISAACSIM_PATH}/kit/kit" "${ISAACSIM_PATH}/apps/omni.isaac.sim.kit" \
        --ext-folder "${ISAACSIM_PATH}/apps" \
        "$@"
fi 