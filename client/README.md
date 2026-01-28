# MCP Client: Scene Generation Interface

This repository contains the client-side implementation for controlling the Scene Generation pipeline.

## 1. Installation & Setup

### 1.1 Python Environment
Install the `sage` Python environment using the provided configuration at `./environment.yml`.

### 1.2 Material Generation Engine
1.  **Matfuse:** Follow the installation guidance at `../matfuse-sd`.
2.  **Configuration:** Update the Matfuse directory path in `./constant.py`.
*   *Note:* Flux is also supported. See `./start_flux_server.sh` (Requires `HF_TOKEN`).

### 1.3 Isaac Sim Installation
1.  **Download:** Download **Isaac Sim 4.2.0** from the [NVIDIA Archives](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html).
2.  **Install:** Extract to your preferred directory (e.g., `~/isaacsim`).
3.  **Link Extension:** Symlink the MCP extension to the Isaac Sim extensions directory to enable server connection:
```bash
ln -s $(realpath ../server/isaacsim/isaac.sim.mcp_extension) ~/isaacsim/exts/isaac.sim.mcp_extension
```

### 1.4 Start MCP Client VLM (Thinking Version)
We utilize **Qwen3-VL-32B-Thinking** hosted with vllm.

```bash
# Download model
hf download Qwen/Qwen3-VL-32B-Thinking --local-dir /tmp/Qwen3-VL-32B-Thinking

# Serve model
cd /tmp
vllm serve Qwen3-VL-32B-Thinking \
    --port 8080 \
    --trust-remote-code \
    --async-scheduling \
    --mm-processor-cache-gb 0 \
    --tensor-parallel-size 8 \
    --reasoning-parser qwen3
```

### 1.5 Configuration
Fill in the API token and URL in `./key.json`.

## 2. Scene Generation Usage

### 2.1 Start Isaac Sim (Background Service)
First, start the Isaac Sim server. Modify the start script if necessary at `./isaac_sim_conda.sh`.
*   Verify the connection: You should see a connection to `localhost:xxxxx` in the output.

### 2.2 Generation Commands
Once Isaac Sim is running, use the following scripts to generate scenes.

**I. Generate from Single Room Descriptions**
```bash
./scripts/generate_from_room_desc.sh
```

**II. Generate from Robot Task Descriptions**
```bash
./scripts/generate_from_robot_task.sh
```

**III. Multi-Room Generation**
```bash
./scripts/generate_multi_rooms.sh
```

*   **Image Conditioning:** All scripts support image conditioning by appending the image path as an input argument.

## 3. Scene Layout-Level Augmentation

For layout-level augmentation (where task-related objects are maintained while the rest of the layout is re-generated):

**Prerequisite:** A base layout is required.

**Run Augmentation:**
```bash
./scripts/generate_scene_augmentation.sh
```
