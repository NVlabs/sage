source ~/.bashrc
cd SAGE

conda activate sage
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CPATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

while true; do echo "Starting at $(date)..."; ./client/isaac_sim_conda.sh --no-window omni.isaac.sim --enable isaac.sim.mcp_extension; echo "Crashed at $(date), restarting..."; done &
cd ./client

## Examples:

python client_generation_room_desc.py \
    --room_desc "A bedroom." \
    --server_paths ../server/layout_wo_robot.py

python client_generation_room_desc.py \
    --room_desc "A medium-sized kitchen." \
    --server_paths ../server/layout_wo_robot.py

python client_generation_room_desc.py \
    --room_desc "A medium-sized rusty, dusty, and abandoned restroom." \
    --server_paths ../server/layout_wo_robot.py

python client_generation_room_desc.py \
    --room_desc "A medium-sized van gogh the starry night style bedroom." \
    --server_paths ../server/layout_wo_robot.py

python client_generation_room_desc.py \
    --room_desc "A living room with a coffee table holding a small toy rubik cube, a student desk positioned away from the coffee table, and a round table with a coke can positioned away from both other tables" \
    --server_paths ../server/layout_wo_robot.py



