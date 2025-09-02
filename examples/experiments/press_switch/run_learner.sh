
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=press_switch \
    --checkpoint_path=first_run \
    --demo_path=/home/kkk/workspace/hil-serl/examples/experiments/press_switch/demo_data/press_switch_20_demos_2025-08-29_22-30-18.pkl \
    --learner \