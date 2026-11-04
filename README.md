<div align="center">

# NanoKV: High-Ratio KV Cache Compression for Autoregressive Video Generation

### Generation-aware KV compression with a frozen video backbone

Jiacheng Lu¹ · Haoyi Zhu² · Sipei Yi¹ · Enze Xie² · Cheng Zhuo¹ · Yu Li¹*

¹ Zhejiang University &nbsp; ² NVIDIA &nbsp; *Corresponding author

[![Project Page](https://img.shields.io/badge/Project-Page-0055AA)](https://jiacheng-lu03.github.io/NanoKV/)

**[🌐 Project](https://jiacheng-lu03.github.io/NanoKV/) · [🚀 Inference](infer/README.md) · [🏋️ Training](train/README.md) · [⚙️ Setup](docs/SETUP.md)**

</div>

## 📖 Introduction

NanoKV compresses historical key–value caches for autoregressive video generation.
It combines shared-center residual coding and low-bit coefficients with
**generation-aware, role-specific K/V fitting**: Key decoders are aligned to
downstream block outputs, while Value codecs are fitted through the deployed
quantization path. The video backbone stays frozen.

**NanoKV-Fast** optionally reuses the same compressed coefficients for
mixed-dimensional attention. Its model-specific configurations balance generation
quality, cache storage, and historical-attention computation.

## ✨ Highlights

- **Three video backbones:** HY-WorldPlay, LingBot-World 2.0, and Matrix-Game 3.5.
- **Two inference modes:** NanoKV for compression and NanoKV-Fast for optional
  attention acceleration.
- **Complete fitting workflow:** collect KV samples, fit residual representations
  and codecs, fit Fast readers, and export inference artifacts.
- **Shared command interface:** model-specific implementations behind common
  training, inference, and evaluation entrypoints.

## 🚀 Quick Start

### 1. Clone and set up one model

```bash
git clone https://github.com/jiacheng-lu03/NanoKV.git
cd NanoKV
```

Follow [environment setup](docs/SETUP.md). Each backbone uses its own Python 3.10
environment; NanoKV and NanoKV-Fast share the environment for that backbone.

### 2. Prepare the backbone and codec

Download the original model checkpoints using the upstream model instructions
linked in the setup guide. For compressed inference, also prepare `codec.pt`;
NanoKV-Fast additionally uses fitted reader files and their manifest.

Codec and reader weights are planned for Hugging Face. Until those downloads
are available, follow the [training guides](train/README.md) to produce them.
The [artifact guide](configs/README.md) describes how to bind local artifacts
to an inference profile.

### 3. Generate a video

| Model | Command name | Inference | Collection and training |
|---|---|---|---|
| HY-WorldPlay | `hy` | [Generate](infer/README.md#hy-worldplay) | [Train](train/hy.md) |
| LingBot-World 2.0 | `lingbot` | [Generate](infer/README.md#lingbot-world-20) | [Train](train/lingbot.md) |
| Matrix-Game 3.5 | `mg` | [Generate](infer/README.md#matrix-game-35) | [Train](train/mg.md) |

Run all guide commands from the repository root. Supply **absolute paths** for
your checkpoints, inputs, and outputs: the launcher runs the native model from
its own integration directory. Options before `--` belong to the launcher;
options after it belong to the model.

```bash
python infer/run.py hy                    # list inference stages
python train/run.py hy                    # list fitting stages
python train/run.py hy fit_value -- --help
```

## 🏋️ Training

The workflow is **collection → residual basis fitting → K/V codec fitting →
packed codec export → optional Fast reader fitting**.

Start with the [training overview](train/README.md), then use the model-specific
guide. They specify input manifests, sample layouts, fitting commands, outputs,
and the transition from training to inference. The [Fast guide](train/fast.md)
starts from a fitted NanoKV codec and ends with a runnable Fast profile.

## 📊 Evaluation

The [inference guide](infer/README.md#baselines) includes Original, KVTC, QVG,
and TurboQuant profiles. Compare videos with identical input images, prompts,
actions, seeds, frame counts, and resolutions using
[frame-aligned metrics](infer/README.md#video-metrics).

## 🗂️ Repository Layout

```text
nanokv/    Shared launch and artifact-binding utilities
models/    Three backbone integrations, each with compression/ and fast/
configs/   Inference profiles, codec budgets, and fixed head allocations
train/     Collection, codec fitting, and Fast reader fitting guides/entrypoints
infer/     Inference guide and entrypoint
eval/      Frame-aligned video metrics
docs/      Environment setup and project page
```

## 🤝 Acknowledgements and Licenses

We build on HY-WorldPlay, LingBot-World, Matrix-Game, and their dependencies.
Included upstream code retains its original license and notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Follow the respective upstream
terms when obtaining and using backbone weights.
