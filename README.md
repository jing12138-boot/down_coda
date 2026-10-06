# SGDS-YOLOE: Down and Feather Component Detection

This directory collects the key model configurations, method implementations, training scripts, ablation configurations, and comparison-model code for SGDS-YOLOE. The framework combines a YOLOE image-text detection head with a P2-P4 detection layout, FeatherP2Fusion, C3k2_StripBlock, and Scale-Adaptive NWD.

All directory and file names use English. This collection contains source code, configuration files, and environment specifications.

## Directory layout

```text
down_code/
|-- README.md
|-- model_configs/
|   |-- baseline/
|   |-- sgds/
|   |-- ablations/
|   |-- comparisons/
|   `-- additional_adapters/
|-- experiment_settings/
|-- environment/
`-- source/
    |-- YOLO26_E/
    |   |-- configs/
    |   |-- tools/
    |   |-- ultralytics/
    |   `-- training and evaluation scripts
    `-- YOLOE_Feather_Comparisons/
        |-- configs/
        |-- reference/
        |-- tests/
        |-- ultralytics/nn/tasks.py
        `-- adapter training and comparison scripts
```

`model_configs/` provides short, readable names for the model graphs. The configurations under `source/` retain the file names referenced by the training scripts. Model YAML files describe network structure; training scripts and `experiment_settings/` specify losses, augmentation, optimization, and evaluation settings.

## Main model configurations

| Model | Configuration |
|---|---|
| YOLOE-26n baseline | [YOLOE26n.yaml](model_configs/baseline/YOLOE26n.yaml) |
| SGDS-YOLOE | [SGDS_YOLOE.yaml](model_configs/sgds/SGDS_YOLOE.yaml) |
| HFP-P234 | [HFP_P234.yaml](model_configs/comparisons/HFP_P234.yaml) |
| NWD-Focal | [NWD_Focal.yaml](model_configs/comparisons/NWD_Focal.yaml) |
| Global NWD | [Global_NWD.yaml](model_configs/comparisons/Global_NWD.yaml) |
| Multipole | [Multipole.yaml](model_configs/comparisons/Multipole.yaml) |

The baseline uses P3-P5 detection outputs. SGDS-YOLOE uses P2-P4 outputs while preserving the deep backbone context path. A YOLOE model YAML can declare `nc: 80` for its grounding prompt slots; the evaluation configuration defines the five component categories. The `n` model scale is selected by the relevant configuration or model-loading path.

## Key method implementations

Paths in this table are relative to `source/YOLO26_E/`.

| Component | File | Main definitions |
|---|---|---|
| Detail-semantic feature fusion | [FeatherP2Fusion.py](source/YOLO26_E/ultralytics/nn/newsAddmodules/FeatherP2Fusion.py) | `FeatherP2Fusion` |
| Directional feature processing | [StripConv_AAAI2026.py](source/YOLO26_E/ultralytics/nn/newsAddmodules/StripConv_AAAI2026.py) | `StripConv`, `Attention`, `StripBlock`, `C3k_StripBlock`, `C3k2_StripBlock` |
| Scale-Adaptive NWD and classification losses | [loss.py](source/YOLO26_E/ultralytics/utils/loss.py) | `wasserstein_loss`, `BboxLoss`, `v8DetectionLoss` |
| Detection and image-text matching head | [head.py](source/YOLO26_E/ultralytics/nn/modules/head.py) | `Detect`, `YOLOEDetect` |
| Base network blocks and text alignment components | [block.py](source/YOLO26_E/ultralytics/nn/modules/block.py) | `C2f`, `C3`, `C3k2`, `SPPF`, `C2PSA`, `BNContrastiveHead`, `Residual`, `SwiGLUFFN` |
| Text encoder construction | [text_model.py](source/YOLO26_E/ultralytics/nn/text_model.py) | Text model construction and encoding |
| Model construction and module registration | [tasks.py](source/YOLO26_E/ultralytics/nn/tasks.py) | `YOLOEModel`, `parse_model` |
| HFP-P234 comparison modules | [HFP_SDP_2025AAAI.py](source/YOLO26_E/ultralytics/nn/newsAddmodules/HFP_SDP_2025AAAI.py) | `HFP`, `SDPFusion`, related feature-pyramid operations |
| Multipole comparison module | [MultipoleAttention_2025ICCV.py](source/YOLO26_E/ultralytics/nn/newsAddmodules/MultipoleAttention_2025ICCV.py) | `C2PSA_MultipoleAttention` and related attention blocks |

FeatherP2Fusion combines shallow P2 details with upsampled P3 visual features using channel and spatial gates. Text embeddings do not directly enter this fusion module. StripBlock performs local and directional depthwise convolutions inside a CSP channel path. Image-text matching occurs in the YOLOE detection head.

Scale-Adaptive NWD is implemented in the loss code, not in a separate network-layer class. Its principal switches are `nwdloss`, `iou_ratio`, `adaptive_nwdloss`, `adaptive_nwd_scale`, and `adaptive_nwd_temperature`. The SGDS settings enable scale adaptation with a transition scale of 32 pixels, a temperature of 8 pixels, and an NWD mixing-weight upper bound of 0.1. Classification uses BCE when `focal_gamma` is zero.

## Training entry points

The following scripts are located in `source/YOLO26_E/`.

| Purpose | Script |
|---|---|
| SGDS-YOLOE three-run training | [run_sgds_yoloe_pycharm.py](source/YOLO26_E/run_sgds_yoloe_pycharm.py) |
| YOLOE-26n three-run comparison | [run_YOLOE26n_triplicate.py](source/YOLO26_E/run_YOLOE26n_triplicate.py) |
| HFP-P234 three-run comparison | [run_HFP_P234_triplicate.py](source/YOLO26_E/run_HFP_P234_triplicate.py) |
| NWD-Focal three-run comparison | [run_NWD_Focal_triplicate.py](source/YOLO26_E/run_NWD_Focal_triplicate.py) |
| Global NWD three-run comparison | [run_Global_NWD_triplicate.py](source/YOLO26_E/run_Global_NWD_triplicate.py) |
| Multipole three-run comparison | [run_Multipole_triplicate.py](source/YOLO26_E/run_Multipole_triplicate.py) |
| Shared comparison execution and aggregation | [run_table4_triplicate_common.py](source/YOLO26_E/run_table4_triplicate_common.py) |
| Configurable training and evaluation | [run_yoloe26_paper.py](source/YOLO26_E/run_yoloe26_paper.py) |
| SGDS training, input checks, and evaluation support | [run_yoloe26_repro.py](source/YOLO26_E/run_yoloe26_repro.py) |
| Grounding trainer | [train_yoloe_grounding.py](source/YOLO26_E/train_yoloe_grounding.py) |
| Training utilities | [train_utils.py](source/YOLO26_E/train_utils.py) |
| Run auditing | [audit_yoloe_run.py](source/YOLO26_E/audit_yoloe_run.py) |

The main protocol uses 120 epochs, an image size of 640, physical batch size 4, nominal batch size 8, and zero data-loader workers. Model selection uses validation mAP50-95. Test evaluation uses the selected checkpoint. Check the selected script's path settings and output directory before execution.

The NWD-Focal comparison has its own loss and augmentation settings: `iou_ratio=0.7`, `focal_gamma=1.5`, `focal_alpha=0.25`, `mosaic=0.25`, and `close_mosaic=20`. These settings are defined in its runner and configuration. SGDS-YOLOE and Global NWD use BCE, `mosaic=0.5`, and `close_mosaic=12`.

## Core ablations

| Group | Configuration in `model_configs/ablations/` | P2 fusion | Directional block | Localization |
|---|---|---|---|---|
| A | `A_YOLOE26n.yaml` | No P2 branch | No | CIoU |
| B | `B_P2Plain.yaml` | Concatenation | No | CIoU |
| C | `C_P2Fusion.yaml` | FeatherP2Fusion | No | CIoU |
| D | `D_P2Strip.yaml` | Concatenation | C3k2_StripBlock | CIoU |
| E | `E_P2FusionStrip.yaml` | FeatherP2Fusion | C3k2_StripBlock | CIoU |
| F | `F_Global_NWD.yaml` | FeatherP2Fusion | C3k2_StripBlock | Fixed 90% CIoU + 10% NWD |
| G | `G_SGDS_YOLOE.yaml` | FeatherP2Fusion | C3k2_StripBlock | Scale-Adaptive NWD |

The configurable training entry point is `run_yoloe26_paper.py`. Its arguments include `--model-config`, `--nwdloss`, `--iou-ratio`, `--adaptive-nwdloss`, `--adaptive-nwd-scale`, `--adaptive-nwd-temperature`, `--focal-gamma`, `--mosaic`, and `--close-mosaic`. Groups A-E disable NWD. Group F enables fixed NWD, while Group G also enables scale adaptation. Changing a YAML file alone does not configure the localization loss.

The matching training-argument files are in `experiment_settings/`: `YOLOE26n.yaml`, `Ablation_B_P2Plain.yaml`, `Ablation_C_P2Fusion.yaml`, `Ablation_D_P2Strip.yaml`, `Ablation_E_P2FusionStrip.yaml`, `Global_NWD.yaml`, and `SGDS_YOLOE.yaml`. Comparison settings for HFP-P234, NWD-Focal, and Multipole are provided there as well.

## Multimodal text interventions

The text-prompt evaluation code supports correct prompts, a fixed shuffled category-text mapping, and zero text embeddings:

- [run_yoloe26_text_ablation.py](source/YOLO26_E/run_yoloe26_text_ablation.py) defines the text-intervention validator, permutation generation, and metric extraction.
- [evaluate_original_sgds.py](source/YOLO26_E/evaluate_original_sgds.py) provides the SGDS checkpoint evaluation entry point.

The evaluation settings include the test split, image size 640, batch size 4, confidence threshold 0.001, and `max_det=300`. The evaluation scripts require external checkpoint and dataset files. No evaluation output is supplied in this directory.

## Additional YOLOE adapter comparisons

`source/YOLOE_Feather_Comparisons/` contains four Python entry points:

- `run_YOLOE5n_Adapted.py`
- `run_YOLOE8n_Adapted.py`
- `run_YOLOE9t_Adapted.py`
- `run_YOLOE11n_Adapted.py`

Each entry point uses three repetitions through `run_comparison_common.py`. The shared implementation is in `adapter_trainer.py`, `experiment_config.py`, `experiment_io.py`, and `experiment_runtime.py`. These are custom combinations of the corresponding visual backbone/neck with a YOLOE detection head. They use P3-P5 outputs and do not incorporate FeatherP2Fusion, C3k2_StripBlock, or Scale-Adaptive NWD. The YOLOv9 adapter uses the `t` size.

Readable model configurations are also provided in `model_configs/additional_adapters/`. The `reference/` files are training-argument configurations, not performance results.

## Environment and use in PyCharm

The scripts target the local Conda interpreter `E:/anaconda/envs/YOLO26/python.exe`. Environment specifications are provided in:

- `environment/requirements-lock.txt`
- `environment/environment-history.yml`
- `environment/conda-explicit.txt`

This is a collection of selected implementation files rather than a complete standalone Ultralytics installation. Use the code within the compatible full project and configure that project in PyCharm. Some framework registration files import additional library modules that are not part of this focused collection. Dataset locations, model weights, text-model assets, and output locations must be configured in the relevant scripts and YAML files. The interpreter and paths are not automatically changed by this collection.

The `configs/down_yoloe.yaml` files contain the five English category names and dataset paths only. The grounding YAML files refer to external phrase-box annotation records. `tools/convert_grounding.py` contains the annotation conversion logic, but the annotations themselves are not included.

## Method references

- YOLOE: Real-Time Seeing Anything. DOI: https://doi.org/10.1109/ICCV51701.2025.02280
- MobileCLIP2: Improving Multi-Modal Reinforced Training. https://arxiv.org/abs/2508.20691
- CBAM: Convolutional Block Attention Module. DOI: https://doi.org/10.1007/978-3-030-01234-2_1
- Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection. DOI: https://doi.org/10.1609/aaai.v40i15.38217
- Going Deeper with Image Transformers (LayerScale). DOI: https://doi.org/10.1109/ICCV48922.2021.00010
- Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation (CIoU). DOI: https://doi.org/10.1109/TCYB.2021.3095305
- Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and a New Benchmark. DOI: https://doi.org/10.1016/j.isprsjprs.2022.06.002
- Focal Loss for Dense Object Detection. DOI: https://doi.org/10.1109/ICCV.2017.324
- HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection. DOI: https://doi.org/10.1609/aaai.v39i7.32740

Existing attribution and license notices in the source files are retained.
