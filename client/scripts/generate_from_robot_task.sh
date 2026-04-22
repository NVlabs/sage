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

python client_generation_robot_task.py \
    --room_type "living room" \
    --robot_type "franka arm" \
    --task_description "In a living room with a coffee table holding a small toy rubik cube and a plate, a student desk positioned away from the coffee table, and a round table with a coke can positioned away from both other tables, the robot must pick up the toy rubik cube from the coffee table and place it on the plate on the same coffee table." \
    --server_paths ../server/layout.py

python client_generation_robot_task.py \
    --room_type "living room" \
    --robot_type "mobile franka" \
    --task_description "In a living room with a coffee table holding a small toy rubik cube, a student desk positioned away from the coffee table, and a round table with a coke can positioned away from both other tables, the robot must pick up the toy rubik cube from the coffee table and place it on the student desk, then navigate to the round table to pick up the coke can and place it on the coffee table." \
    --server_paths ../server/layout.py


python client_generation_robot_task.py \
    --room_type "living room" \
    --robot_type "mobile franka" \
    --task_description "In a living room with a coffee table holding a small toy rubik cube, a student desk positioned away from the coffee table, and a round table with a coke can positioned away from both other tables, the robot must navigate to the round table to pick up the coke can and then place it on the coffee table." \
    --server_paths ../server/layout.py

python client_generation_robot_task.py \
    --room_type "bedroom" \
    --robot_type "mobile franka" \
    --task_description "In a bedroom, the robot must pick up the water bottle from the nightstand and place it on the desk." \
    --server_paths ../server/layout.py

python client_generation_robot_task.py \
    --room_type "office" \
    --robot_type "mobile franka" \
    --task_description "In an office with a desk holding a coffee mug and document folder with an office chair in front, a bookshelf with a stapler against the wall, a filing cabinet with a tape dispenser positioned away from the desk and bookshelf, and a side table near the desk, the robot must pick up the coffee mug from the desk and place it on the side table." \
    --server_paths ../server/layout.py

