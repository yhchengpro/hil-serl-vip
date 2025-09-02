export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python /home/kkk/workspace/hil-serl/examples/train_rlpd.py "$@" \
    --exp_name=press_switch \
    --checkpoint_path=/home/kkk/workspace/hil-serl/examples/experiments/press_switch/first_run \
    --actor \
