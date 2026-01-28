source ~/.bashrc
cd SAGE

conda activate sage
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CPATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

while true; do echo "Starting at $(date)..."; ./client/isaac_sim_conda.sh --no-window omni.isaac.sim --enable isaac.sim.mcp_extension; echo "Crashed at $(date), restarting..."; done &
cd ./client


layout_id=layout_xxxxxxxx # to be filled
python client_generation_scene_aug.py \
    --base_layout_dict_path ../server/results/${layout_id}/${layout_id}.json \
    --from_task_required_objects \
    --server_paths ../server/layout_wo_robot_scene_aug.py

