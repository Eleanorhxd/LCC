

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/mimic_cxr}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/save}"

dataset="mimic_cxr"
annotation="${DATA_ROOT}/annotation.json"
base_dir="${DATA_ROOT}/images"

version="v2_latentmorph_mimic_optimized"
savepath="$SAVE_ROOT/$dataset/$version"


if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u "$REPO_ROOT/train.py" \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 14 \
    --val_batch_size 12 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 150 \
    --min_new_tokens 80 \
    --max_new_tokens 150 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 30 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    \
    --use_latent_morph True \
    --latent_morph_d_condenser 512 \
    --latent_morph_d_latent 256 \
    --latent_morph_check_every 16 \
    --latent_morph_max_triggers 3 \
    --latent_morph_min_trigger_position 24 \
    --latent_morph_num_control_tokens 4 \
    --latent_morph_control_strength 1.0 \
    --latent_morph_loss_weight 0.05 \
    \
    --latent_morph_use_vision_reasoning True \
    --latent_morph_vision_reasoning_steps 3 \
    --latent_morph_vision_reasoning_heads 8 \
    \
    --latent_morph_use_cross_modal True \
    --latent_morph_alignment_temp 0.1 \
    \
    2>&1 | tee -a ${savepath}/log.txt
