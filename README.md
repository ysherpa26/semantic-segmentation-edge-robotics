# Semantic Segmentation for Edge Robotics (BudBreak Innovations)

Reproducing and training SegNeXt-B for semantic segmentation on ADE20K, as a step toward edge deployment on autonomous agricultural robots. Fellowship project at BudBreak Innovations, presented at the NYU SPS x Cornell Tech x NewLab Innovation Showcase (April 2026).

## Problem

BudBreak Innovations deploys autonomous robots in vineyards. As a robot drives the rows, it captures camera images and needs to identify what it sees (plants, disease, soil, structures) to help growers make better decisions. Because vineyards often have no internet or cell service, inference must run on the robot's onboard computer (NVIDIA Jetson Orin) in real time, so the model must be both accurate and lightweight.

## Why SegNeXt

SegNeXt uses convolutional attention rather than transformers, achieving strong accuracy with far fewer parameters, making it well-suited for edge devices.

| Model | Type | Params | Published mIoU | Edge-friendly |
|---|---|---|---|---|
| DeepLabV3+ | CNN | 63M | ~45% | Moderate |
| SegFormer-B0 | Transformer | 4M | 37.4% | Moderate |
| SegNeXt-T | Conv attention | 4M | 41.1% | Yes |
| SegNeXt-B | Conv attention | 28M | 48.5% | Yes |

## Training setup

- **Framework:** MMSegmentation (PyTorch)
- **Model:** SegNeXt-B (MSCAN backbone, embed dims [64, 128, 320, 512], LightHamHead decoder)
- **Pretrained weights:** `mscan_b_20230227-3ab7d230.pth` (OpenMMLab)
- **Dataset:** ADE20K (150-class scene parsing, 20K training / 2K validation images)
- **Optimizer:** AdamW, lr=3e-5, weight_decay=0.01, cosine warmup then polynomial decay
- **Training:** 80K iterations, batch size 4, input 512x512, checkpoint every 8K iters
- **Infrastructure:** Google Cloud Vertex AI, NVIDIA Tesla T4, n1-standard-16, 100GB SSD

## The debugging story

The first run hit 5.87% mIoU. Loss was dropping (3.67 to 0.35) but the model only used about 5 of 150 classes and mislabeled cars as "sink" and "cabinet." I worked through it systematically:

1. Tested on training data (6.51% mIoU), ruling out a data pipeline bug
2. Added a pretrained checkpoint and a learning rate schedule, no improvement
3. Visualized predictions and confirmed the model was stuck on ~5 classes
4. **Root cause:** the wrong pretrained checkpoint was loaded (mscan_t weights into the mscan_b model), so zero backbone weights transferred

After loading the correct mscan_b checkpoint, the model learned all 150 classes.

## Results

mIoU on ADE20K validation set by checkpoint:

| Iterations | mIoU | aAcc | mAcc |
|---|---|---|---|
| 8K | 12.05% | 61.06% | 16.72% |
| 16K | 20.61% | 65.39% | 29.40% |
| 24K | 24.68% | 71.93% | 33.25% |
| 32K | 26.78% | 70.61% | 36.63% |
| 40K | 25.55% | 66.26% | 34.15% |
| 48K | 30.34% | 74.51% | 40.03% |
| 56K | 30.93% | 73.75% | 41.70% |
| **64K** | **33.27%** | **75.29%** | **44.01%** |
| 72K | 33.00% | 75.27% | 43.35% |
| 80K | ~33.9% | — | — |

Best checkpoint: 64K iterations, 33.27% mIoU. Published SegNeXt-B target: 48.5%.

Strong classes: sky (85.9%), ceiling (71.0%), car (66.4%), tree (65.2%), person (64.2%). Zero or near-zero on rare classes (bicycle, boat, flag, etc.), which is expected at this training scale before vineyard-specific fine-tuning.

### Prediction progression (same validation image at 8K, 40K, and 80K iterations)

Ground truth (left) vs. model prediction (right):

![8K iterations](results/vis_samples/prediction_8k_iters.png)
![40K iterations](results/vis_samples/prediction_40k_iters.png)
![80K iterations](results/vis_samples/prediction_80k_iters.png)

## Repo contents

```
configs/
  segnext_ade20k.py        Training config (MMSeg format)
  segnext_ade20k_eval.py   Evaluation config
results/
  eval/                    Eval metrics JSON
  vis_samples/             Prediction images at 8K, 40K, 80K iterations
.gitignore                 Excludes checkpoints, dataset, credentials
```

Not included: model checkpoints (.pth), the ADE20K dataset, Google Cloud credentials. See [MIT ADE20K](https://ade20kchallenge.csail.mit.edu/) for the dataset.

## Setup

```bash
# Install MMSegmentation
pip install -U openmim
mim install mmengine "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"

# Download ADE20K and place at ./data/ade20k/
# Download pretrained backbone:
# https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth

# Train
python tools/train.py configs/segnext_ade20k.py

# Evaluate
python tools/test.py configs/segnext_ade20k_eval.py checkpoints/iter_80000.pth
```

## What I would do differently

- Validate checkpoint architecture compatibility before starting a long training run. A one-line check (`model.load_state_dict(..., strict=False)` and inspecting missing keys) would have caught the mscan_t/mscan_b mismatch immediately.
- The gap between 33.27% (my best) and 48.5% (published) is partly explained by evaluating at 80K rather than the paper's 160K iterations. Extending training is the clearest next step.
- Infrastructure time dominated the project early on. I spent significant effort on Docker platform issues, exit codes, and GCS pipeline setup before any model ran. Containerizing with multi-arch builds (`docker buildx`) from the start would have saved several days.

## Next steps

- Extend training to 160K iterations (published target: 48.5% mIoU)
- Fine-tune on BudBreak's vineyard dataset
- Export to ONNX and TensorRT for the Jetson Orin
- Field deployment, Long Island vineyards (May 2026)

## Notes on code and permissions

This was a fellowship project at BudBreak Innovations. The configs and results here are published with permission. Internal GCS bucket names and project IDs have been removed from configs.

## Acknowledgments

Ertai Liu (BudBreak Innovations), Cornell Tech, NYU SPS MSMA.
