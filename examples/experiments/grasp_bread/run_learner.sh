export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=grasp_bread \
    --checkpoint_path=first_run \
    --demo_path=/home/kkk/workspace/hil-serl/examples/experiments/grasp_bread/demo_data/grasp_bread_20_demos_2025-08-25_21-28-05.pkl \
    --learner \