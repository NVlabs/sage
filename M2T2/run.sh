python demo.py eval.checkpoint=m2t2.pth \
    eval.data_dir=/home/hongchix/codes/simgen-mcp/server/results/layout_4762e97d/m2t2_data/19 \
    eval.mask_thresh=0.4 \
    eval.num_runs=5

python demo.py eval.checkpoint=m2t2.pth \
    eval.data_dir=/home/hongchix/codes/simgen-mcp/server/results/layout_a2f73707/m2t2_data/19 \
    eval.mask_thresh=0.4 \
    eval.num_runs=5

python demo.py eval.checkpoint=m2t2.pth \
    eval.data_dir=/home/hongchix/codes/simgen-mcp/server/results/layout_a2f73707/m2t2_data_single/19 \
    eval.mask_thresh=0.4 \
    eval.num_runs=1