
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/iu_xray}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/save}"

dataset="iu_xray"
annotation="${DATA_ROOT}/annotation.json"
base_dir="${DATA_ROOT}/images"

savepath="$SAVE_ROOT/$dataset/$version"
delta_file="${DELTA_FILE:-$savepath/checkpoints/your_checkpoint.pth}"


python -u "$REPO_ROOT/train.py" \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 16 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    \
    --use_latent_morph True \
    --latent_morph_d_condenser 512 \
    --latent_morph_d_latent 256 \
    --latent_morph_check_every 16 \
    --latent_morph_max_triggers 3 \
    --latent_morph_min_trigger_position 32 \
    --latent_morph_num_control_tokens 4 \
    --latent_morph_control_strength 1.0 \
    \
    --latent_morph_use_vision_reasoning True \
    --latent_morph_vision_reasoning_steps 3 \
    --latent_morph_vision_reasoning_heads 8 \
    \
    --latent_morph_use_cross_modal True \
    --latent_morph_alignment_temp 0.07 \
    \
    2>&1 | tee -a ${savepath}/test_log.txt

