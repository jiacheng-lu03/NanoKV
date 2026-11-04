# Collection and training

NanoKV keeps the video backbone frozen. Codec fitting and Fast reader fitting
are separate stages. Start with the guide for your backbone:

| Model | Codec fitting | Fast reader fitting |
|---|---|---|
| HY-WorldPlay | [Complete guide](hy.md) | [HY readers](fast.md#hy-worldplay) |
| LingBot-World 2.0 | [Complete guide](lingbot.md) | [LingBot readers](fast.md#lingbot-world-20) |
| Matrix-Game 3.5 | [Complete guide](mg.md) | [MG readers](fast.md#matrix-game-35) |

## Before starting

1. Complete [environment setup](../docs/SETUP.md) and obtain the original model.
2. Prepare fitting and validation input images, prompts, and model-native actions.
   Split by image; keep every action trajectory from an image in the same split.
   Use separate images for final evaluation.
3. Choose an absolute working directory outside the source checkout.
4. Run commands from the repository root in the selected model environment.

All guides explicitly connect the output of one stage to the input of the next.
Counts in commands are consistency gates, not automatic subsampling: adjust
them when deliberately changing the dataset size.

## Workflow and outputs

| Stage | Input | Output |
|---|---|---|
| Collection | Backbone, images, prompts, actions | Paired KV samples and split manifests |
| Representation fitting | Training KV samples | Shared-center residual PCA bases |
| Value fitting | Bases and train/validation samples | Quantization-aware Value codec |
| Key alignment (HY/LingBot) | Key bases and training trajectories | Block-output-aligned Key decoder |
| Codec export | Fitted components | Packed `codec.pt` |
| Optional Fast fitting | Frozen codec and attention samples | Per-width reader experts |
| Fast export | Experts and fixed head allocation | Readers, manifest, and inference profile |

The 32× dimensional/bitwidth settings use ranks **256 / 640 / 384** for
HY / LingBot / MG. All three use 64 shared centers, two clustering iterations,
and INT4 coefficients. Value fitting uses 12,000 AdamW updates, batch size 512,
and learning rate `1e-4`. Effective storage ratios additionally account for
centers, indices, scales, and protected history.

See [paper_settings.json](../configs/paper_settings.json) for configuration
values and the model guides for their native arguments.

## Command conventions

```bash
python train/run.py hy                          # list stages
python train/run.py hy fit_value -- --help       # native fitting help
python train/run.py lingbot fit_fast -- --help
python train/run.py mg fit_fast -- --help
```

Launcher options (`--profile`, `--env`, `--dry-run`) go before `--`.
Native model arguments go after it. Use absolute paths for data and outputs
because the launcher changes into the model integration directory.

A launcher `--dry-run` displays the resolved command without starting training.
It does not validate the dataset or load model weights. After training, use
`python -m nanokv.prepare_profile` to bind your new codec checksum; the
published-weight profiles are bound to the published artifacts.

## Check a completed run

- Collection: all requested layers and splits are present; MG's manifest helper
  validates the files before fitting.
- Value fitting: the selected checkpoint and validation report exist for every layer.
- Key fitting: all required layer decoders have been exported.
- Packing: `codec.pt` exists and profile creation succeeds.
- Fast: reader assembly succeeds with no missing experts, then profile binding
  confirms that the codec and reader manifest match.
- Inference: generate a video and use the [evaluation guide](../infer/README.md#video-metrics)
  to compare it with an Original run using identical inputs.
