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

pose_group_idx=scene_level_pose_aug_test

python augment/scene_level_pose_aug.py \
    --layout_id $layout_id \
    --room_id $room_id \
    --pose_aug_name $pose_group_idx \
    --aug_num 3


python augment/scene_level_pose_aug_compose.py \
    --layout_id $layout_id \
    --room_id $room_id \
    --pose_aug_name $pose_group_idx
