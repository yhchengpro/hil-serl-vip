export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=open_drawer \
    --checkpoint_path=third_run \
    --demo_path=../../demo_data/open_drawer_20_demos_2025-08-12_11-44-24.pkl \
    --learner \