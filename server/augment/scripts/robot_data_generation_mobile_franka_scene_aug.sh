source ~/.bashrc
cd SAGE

conda activate sage
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CPATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

while true; do echo "Starting at $(date)..."; ./client/isaac_sim_conda.sh --no-window omni.isaac.sim --enable isaac.sim.mcp_extension; echo "Crashed at $(date), restarting..."; done &
cd ./server

layout_id=layout_xxxxxxxx # to be filled
layout_id_original=layout_yyyyyyyy # to be filled, base scene layout id

scene_save_dir=./results/${layout_id}/
cp ${scene_save_dir}/${layout_id}.json ${scene_save_dir}/${layout_id}_original.json

# Step 1: Scene layout-level augmentation object mapping for task-related objects
python augment/scene_aug_mapping_task.py \
    --layout_id $layout_id \
    --layout_id_original $layout_id_original

# Step 2: Ensure the navigation feasibility
python isaaclab/correct_mobile_franka.py \
    --layout_id $layout_id

# Step 3: Pose augmentation
pose_aug_name=pose_aug_0
python augment/pose_aug_mm_from_layout_with_task.py \
    --save_dir_name ${pose_aug_name} \
    --num_samples 8 \
    --layout_id $layout_id

# Step 4: Data generation
python isaaclab/data_generation_mobile_franka_mobile_manipulation_with_pose_aug.py \
    --enable_cameras \
    --headless \
    --visualize_figs \
    --post_fix motion_plan_${layout_id}_${pose_aug_name} \
    --num_demos 16 \
    --layout_id ${layout_id} \
    --pose_aug_name ${pose_aug_name}

# Step 5: Step Decomposition
python isaaclab/data_generation_mobile_manipulation_v7_replay.py \
    --post_fix motion_plan_${layout_id}_${pose_aug_name} \
    --verbose \
    --split_by_stages