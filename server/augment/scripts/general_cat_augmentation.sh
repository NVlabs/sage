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
room_id=room_xxxxxxxx # to be filled

type_group_idx=00
pose_group_idx=00

type_aug_name=scene_level_type_aug_test_${type_group_idx}

python augment/scene_level_type_aug.py \
    --layout_id $layout_id \
    --room_id $room_id \
    --aug_name $type_aug_name \
    --aug_num 4

python augment/scene_level_type_aug_settle.py \
    --layout_id $layout_id \
    --room_id $room_id \
    --type_aug_name $type_aug_name \
    --pose_aug_name aug_pose_group_${pose_group_idx}

python augment/scene_level_type_aug_compose.py \
    --layout_id $layout_id \
    --room_id $room_id \
    --type_aug_name $type_aug_name \
    --pose_aug_name aug_pose_group_${pose_group_idx}

python augment/scene_level_type_aug_visualize_room_bpy_frames_circular_cam_sampling_from_file.py \
    --layout_id $layout_id \
    --room_id $room_id \
    --type_aug_name $type_aug_name \
    --pose_aug_name aug_pose_group_${pose_group_idx}