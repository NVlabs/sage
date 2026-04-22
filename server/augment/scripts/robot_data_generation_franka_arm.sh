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
type_aug_name=aug_type_from_layout_with_task_test

## I part
python augment/type_aug_from_layout_with_task.py \
    --layout_id $layout_id \
    --aug_name $type_aug_name \
    --aug_num 4

pose_group_idx=00

python augment/type_aug_with_pose_aug_from_layout_with_task.py \
    --layout_id $layout_id \
    --type_aug_name $type_aug_name \
    --pose_aug_name aug_pose_group_${pose_group_idx} \
    --aug_num 4

# option 1: 
# single type candidate
type_idx=0
python isaaclab/data_generation_franka_pnp_type_aug_with_pose_aug.py \
    --num_envs 4 \
    --layout_id $layout_id \
    --type_aug_name $type_aug_name \
    --type_candidate_id ${layout_id}_${type_aug_name}_mixed${type_idx} \
    --pose_aug_name aug_pose_group_${pose_group_idx} \
    --enable_cameras \
    --headless

# option 2:
# all type candidates
# Set the start and end indices for type candidates

start_type_idx=0  # Change this to your desired start index
end_type_idx=64    # Change this to your desired end index (exclusive)

# Loop through type_idx from start_type_idx to end_type_idx-1
for type_idx in $(seq $start_type_idx $((end_type_idx-1))); do
    echo "Processing type_idx: $type_idx"
    
    python isaaclab/data_generation_franka_pnp_type_aug_with_pose_aug.py \
        --num_envs 1 \
        --layout_id $layout_id \
        --type_aug_name $type_aug_name \
        --type_candidate_id ${layout_id}_${type_aug_name}_mixed${type_idx} \
        --pose_aug_name aug_pose_group_${pose_group_idx} \
        --enable_cameras \
        --headless
done
