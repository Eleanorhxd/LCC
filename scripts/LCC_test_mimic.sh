#!/bin/bash

# ============================================================================
# LatentMorph Enhanced 测试脚本（MIMIC-CXR，对应 latentmorph_train_mimic.sh）
#
# 使用前设置 DELTA_FILE 为 checkpoint，或改下方默认值
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/mimic_cxr}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/save}"

dataset="mimic_cxr"
annotation="${DATA_ROOT}/annotation.json"
base_dir="${DATA_ROOT}/images"

version="v1_latentmorph_enhanced"
savepath="$SAVE_ROOT/$dataset/$version"
# 训练完成后在 savepath/checkpoints/ 下选择权重，或通过环境变量 DELTA_FILE 指定
delta_file="${DELTA_FILE:-$savepath/checkpoints/your_checkpoint.pth}"

echo "=========================================="
echo "LatentMorph Enhanced Test (MIMIC-CXR)"
echo "=========================================="
echo "Dataset: $dataset"
echo "Checkpoint: $delta_file"
echo "=========================================="

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
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
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

echo "=========================================="
echo "LatentMorph test completed!"
echo "=========================================="
