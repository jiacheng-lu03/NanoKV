# Environment setup

Run these commands from the NanoKV repository root. Use one Python 3.10
environment per backbone. Both variants of a model share that environment.
Install a CUDA-capable NVIDIA driver and the CUDA build toolkit needed by
FlashAttention. The reference runs used NVIDIA A100 GPUs.

| Model | PyTorch / CUDA | torchvision | Transformers | NumPy | FlashAttention |
|---|---|---|---|---|---|
| HY | 2.6.0 / 12.4 | 0.21.0 | 4.56.0 | 1.26.4 | 2.8.3 |
| LingBot | 2.4.0 / 12.1 | 0.19.0 | 4.51.3 | 1.26.4 | 2.8.3.post1 |
| MG | 2.11.0 / 12.8 | 0.26.0 | 5.15.0 | 2.2.6 | Not required |

These are the runtime versions used for the release checks. The constraints
below keep installation from upgrading the main numerical dependencies.

## HY-WorldPlay

```bash
conda create -n nanokv-hy python=3.10 -y
conda activate nanokv-hy
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -c configs/environment-hy.txt \
  -r models/HY-WorldPlay/compression/requirements.txt
```

Download [HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
and the action-model checkpoint following
[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay).
`--model_path` points to the base-model directory; `--action_ckpt` points to
the action model's `model.safetensors` file.

## LingBot-World 2.0

```bash
conda create -n nanokv-lingbot python=3.10 -y
conda activate nanokv-lingbot
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
python -m pip install -c configs/environment-lingbot.txt \
  -r models/LingBot-World-2.0/compression/requirements.txt
```

Download the **LingBot-World v2 14B causal-fast** model using the
[upstream instructions](https://github.com/Robbyant/lingbot-world).
`--ckpt_dir` points to the complete checkpoint directory, including the text
encoder and VAE. Use the causal-fast weights with the supplied profiles.

## Matrix-Game 3.5

```bash
conda create -n nanokv-mg python=3.10 -y
conda activate nanokv-mg
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c configs/environment-mg.txt \
  -r models/Matrix-Game-3.5/compression/requirements.txt
```

Follow [Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) to obtain the
Matrix-Game 3.5 first-person checkpoint, Wan2.2-TI2V-5B (including its UMT5
tokenizer), and DA3NESTED-GIANT-LARGE-1.1. The inference guide specifies all
four paths explicitly.

## Check the installation

Within the selected environment:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'
python -m pip check
python infer/run.py hy generate -- --help  # use lingbot or mg in its environment
```

Use `CUDA_VISIBLE_DEVICES=1` before a command to select physical GPU 1; the
process sees it as `cuda:0`. Run one model per GPU unless you have checked the
available memory. LingBot Key fitting instead accepts the physical GPU number
through its native `--gpu` argument.

FlashAttention build errors usually require matching the CUDA toolkit and
PyTorch installation before rebuilding the extension. Keep the model
environments separate: their Transformers and PyTorch versions differ.
