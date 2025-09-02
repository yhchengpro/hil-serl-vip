export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python /home/kkk/workspace/hil-serl/examples/train_rlpd.py "$@" \
    --exp_name=grasp_bread \
    --checkpoint_path=/home/kkk/workspace/hil-serl/examples/experiments/grasp_bread/first_run \
    --actor \
    --eval_checkpoint_step=75000 \
    --eval_n_trajs=5 \