# Reason Before You Generate: Latent Consistency Control for Radiology Report Generation


## Requirements

Python 3.10+, CUDA, and a recent PyTorch build are recommended.

Main dependencies (pin versions to match your CUDA stack):

- `torch`, `torchvision`
- `lightning`
- `transformers`
- `peft` (optional, for LoRA)
- `Pillow`, `numpy`

---

## Repository layout (core)

```
LCC/
├── train.py                      
├── dataset/
│   ├── data_module.py       
│   └── data_helper.py       
├── models/
│   └── model.py              
├── lightning_tools/         
└── scripts/               
```

---

## Data preparation

1. Prepare a JSON annotation file; pass its path with `--annotation`.
2. Set `--base_dir` to the image root; each sample’s `image_path` entries are relative to that root.

With `--dataset` set to `iu_xray` or `mimic_cxr`, different report cleaning rules apply (see `dataset/data_helper.py`).

data paths:

- MIMIC-CXR: `./data/mimic_cxr/annotation.json`, images under `./data/mimic_cxr/images/`
- IU-X-ray: point `DATA_ROOT` in the shell scripts to e.g. `./data/iu_xray/`

---

## Pretrained weights

- **Vision:** default `microsoft/swin-base-patch4-window7-224` (override with `--vision_model`).
- **Language:** default local folder `./pretrained/llama-2-7b-chat-hf` (override with `--llama_model` to a local path or a Hugging Face model id; respect the model license).
- Paths for CheXbert / BERT / RadGraph used in evaluation are listed as relative `./checkpoints/...` in `DEFAULT_ARGS` inside `models/model.py`; adjust to your layout or wire equivalent paths in code.

---

## Training and testing

From the repository root (adapt paths and GPU count):

```bash
python train.py \
  --dataset mimic_cxr \
  --annotation ./data/mimic_cxr/annotation.json \
  --base_dir ./data/mimic_cxr/images \
  --llama_model ./pretrained/llama-2-7b-chat-hf \
  --savedmodel_path ./save/mimic/run1 \
  --devices 1 \
  --max_epochs 30
```

Validation only:

```bash
python train.py --validate  
```

Test split (usually load trained weights):

```bash
python train.py --test \
  --delta_file ./save/mimic/run1/checkpoints/your_checkpoint.pth \
  # ...same args as above...
```

Resume from a Lightning checkpoint:

```bash
python train.py --ckpt_file ./save/.../last.ckpt ...
```
