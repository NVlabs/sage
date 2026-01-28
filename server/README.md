# MCP Server: Scene Generation Backend

This repository contains the server-side implementation for the MCP Scene Generation pipeline.

## 1. Preparation & Setup

### 1.1 Start the TRELLIS Server
Refer to the startup script of trellis. We hosted with docker on remote cluster.
```bash
./start_trellis_server.sh
```
Huggingface token `HF_TOKEN` is needed for ckpt downloading.


### 1.2 Start the Vision-Language Model (VLM)
We utilize **Qwen3-VL** hosted with vllm.

**Download and Serve:**
```bash
# Download model
hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir /tmp/Qwen3-VL-30B-A3B-Instruct

# Serve model
cd /tmp
vllm serve Qwen3-VL-30B-A3B-Instruct \
    --port 8080 \
    --max-model-len 32768 \
    --async-scheduling \
    --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}' \
    --mm-processor-cache-gb 0
```

### 1.3 Start the General LLM
We utilize **gpt-oss-120b** hosted with vllm.

**Serve Model:**
```bash
vllm serve openai/gpt-oss-120b --port 8080 --tensor-parallel-size 4 --async-scheduling
```

### 1.4 Configuration
Populate `key.json` with the necessary API keys and URLs and update `MODEL_DICT` if utilizing different model types.

## 2. Generation

For detailed usage instructions on triggering generations, please refer to the **`../client`** directory documentation.

## 3. Augmentation

### 3.1 General Pose Augmentation
To perform pose augmentation for small, on-top objects within a scene, use the following script:
```bash
./augment/scripts/general_pose_augmentation.sh
```

### 3.2 General Object Category-Level Augmentation
To perform category-level augmentation for objects within a scene, use the following script:
```bash
./augment/scripts/general_cat_augmentation.sh
```

## 4. Robot Data Generation

### 4.1 Prerequisites
Ensure **IsaacLab** and **M2T2** are installed. Refer to their respective installation guides at `../IsaacLab` and `../M2T2`.

### 4.2 Static Franka Arm Tasks
For object category-level augmentation applied to robot data generation:
```bash
./augment/scripts/robot_data_generation_franka_arm.sh
```

### 4.3 Mobile Franka Tasks
This pipeline supports scene layout-level augmentation for robot data generation. Currently, it supports automatic data generation for one round of **navigation + pick-and-place**.

**Option A: Pose Augmentation Only (Task-Related Objects)**
```bash
./augment/scripts/robot_data_generation_mobile_franka.sh
```

**Option B: Scene Layout-Level Augmentation**
1.  **Generate Layout Augmentation:** Refer to the `../client/` README for layout generation instructions.
2.  **Generate Data with Pose Augmentation:**
```bash
./augment/scripts/robot_data_generation_mobile_franka_scene_aug.sh
```

## 5. Policy Training

For policy training instructions, please refer to the `../robomimic` documentation and use the generated HDF5 data.
