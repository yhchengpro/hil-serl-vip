export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python train_rlpd.py \
    --exp_name="$1" \
    --checkpoint_path=first_run \
    --actor \
    --eval_checkpoint_step="$1" \
    --show_demo=True \