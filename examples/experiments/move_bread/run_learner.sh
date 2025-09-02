
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=move_bread \
    --checkpoint_path=first_run \
    --demo_path=/home/kkk/workspace/hil-serl/examples/experiments/move_bread/demo_data/move_bread_20_demos_2025-09-01_01-16-44.pkl \
    --learner \