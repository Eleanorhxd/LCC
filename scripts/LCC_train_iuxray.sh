#!/bin/bash

# ============================================================================
# LatentMorph Enhanced 训练脚本：放射学报告生成 + 双向潜在推理
# 
# 增强功能：
# - 视觉潜在空间推理（VisionLatentReasoner）
# - 跨模态对齐（CrossModalAligner）
# - 增强一致性检测（EnhancedImageTextMatcher）
# - 智能控制信号注入
# 
# 核心思想：
# - 在报告生成过程中，不仅在文本空间推理，还在视觉潜在空间推理
# - 跨模态对齐确保报告与图像语义一致
# - 多步视觉推理提供更精确的一致性检测
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/iu_xray}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/save}"

dataset="iu_xray"
annotation="${DATA_ROOT}/annotation.json"
base_dir="${DATA_ROOT}/images"

version="v1_latentmorph_enhanced"
savepath="$SAVE_ROOT/$dataset/$version"

echo "=========================================="
echo "LatentMorph Enhanced Training"
echo "=========================================="
echo "Dataset: $dataset"
echo "Save path: $savepath"
echo ""
echo "Enhanced Features:"
echo "  ✓ Vision Latent Space Reasoning"
echo "  ✓ Cross-Modal Alignment"
echo "  ✓ Multi-Step Vision Reasoning"
echo "=========================================="

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
    --batch_size 8 \
    --val_batch_size 12 \
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
    --max_epochs 15 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    \
    --use_latent_morph True \
    --latent_morph_d_condenser 512 \
    --latent_morph_d_latent 256 \
    --latent_morph_check_every 16 \
    --latent_morph_max_triggers 3 \
    --latent_morph_min_trigger_position 32 \
    --latent_morph_num_control_tokens 4 \
    --latent_morph_control_strength 1.0 \
    --latent_morph_loss_weight 0.1 \
    \
    --latent_morph_use_vision_reasoning True \
    --latent_morph_vision_reasoning_steps 3 \
    --latent_morph_vision_reasoning_heads 8 \
    \
    --latent_morph_use_cross_modal True \
    --latent_morph_alignment_temp 0.07 \
    \
    2>&1 |tee -a ${savepath}/log.txt

echo "=========================================="
echo "LatentMorph Enhanced training completed!"
echo "Checkpoint saved to: $savepath"
echo "=========================================="
