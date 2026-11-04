# Paper Results

[Back to the project overview](../README.md#results)

These values reproduce the main comparison in the NanoKV paper. NanoKV, NanoKV-Fast, and KVTC use the nominal 32× coefficient-compression setting; QVG and TurboQuant use their reported baseline configurations.

## Evaluation protocol

- **Quality:** 100 VBench-I2V images × four camera actions, yielding 400 videos per method and backbone. Inputs and generation settings are matched within each backbone.
- **PSNR, SSIM, and LPIPS:** comparison with the matched Original rollout, measuring fidelity to the uncompressed model.
- **VBench:** mean of five dimensions—subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality.
- **Total KV:** storage including method-specific overhead. Effective KV compression is Original storage divided by method storage, computed before rounding the displayed GiB values.
- **E2E speedup:** Original generation latency divided by method latency, measured on a single NVIDIA A100. Original latency is given separately for each backbone.

## HY-WorldPlay

Original end-to-end latency: **437.63 s**.

| Method | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | VBench ↑ | Total KV ↓ (GiB) | Effective KV compression ↑ | E2E speedup ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | — | — | — | 0.8129 | 20.96 | 1.00× | 1.00× |
| QVG | 17.58 | 0.5379 | 0.3887 | 0.7744 | 3.14 | 6.69× | 0.86× |
| TurboQuant | 17.95 | 0.5459 | 0.3523 | 0.7918 | 4.18 | 5.01× | 0.90× |
| KVTC (32×) | 18.56 | 0.5796 | 0.2900 | 0.7936 | 1.55 | 13.50× | 0.92× |
| NanoKV | 19.56 | 0.6152 | 0.2399 | 0.8057 | 0.80 | 26.09× | 0.96× |
| NanoKV-Fast | 19.04 | 0.5931 | 0.2787 | 0.7981 | 1.02 | 20.53× | 1.06× |

## Matrix-Game 3.5

Original end-to-end latency: **306.70 s**.

| Method | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | VBench ↑ | Total KV ↓ (GiB) | Effective KV compression ↑ | E2E speedup ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | — | — | — | 0.8293 | 11.48 | 1.00× | 1.00× |
| QVG | 16.52 | 0.5150 | 0.4115 | 0.7782 | 2.89 | 3.97× | 0.91× |
| TurboQuant | 16.73 | 0.5314 | 0.3867 | 0.7951 | 2.29 | 5.01× | 0.91× |
| KVTC (32×) | 18.83 | 0.6347 | 0.2380 | 0.8302 | 1.24 | 9.28× | 0.57× |
| NanoKV | 19.08 | 0.6377 | 0.2425 | 0.8273 | 0.80 | 14.26× | 0.92× |
| NanoKV-Fast | 18.79 | 0.6280 | 0.2498 | 0.8280 | 0.96 | 11.94× | 1.04× |

## LingBot-World 2.0

Original end-to-end latency: **132.40 s**.

| Method | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | VBench ↑ | Total KV ↓ (GiB) | Effective KV compression ↑ | E2E speedup ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | — | — | — | 0.8254 | 20.71 | 1.00× | 1.00× |
| QVG | 20.46 | 0.6193 | 0.2310 | 0.8256 | 4.30 | 4.81× | 0.87× |
| TurboQuant | 19.06 | 0.5650 | 0.2890 | 0.8235 | 4.59 | 4.51× | 0.87× |
| KVTC (32×) | 20.10 | 0.6150 | 0.2431 | 0.8228 | 4.73 | 4.38× | 0.79× |
| NanoKV | 20.81 | 0.6352 | 0.2156 | 0.8252 | 1.46 | 14.18× | 0.93× |
| NanoKV-Fast | 20.68 | 0.6314 | 0.2183 | 0.8249 | 1.78 | 11.62× | 0.98× |

## NanoKV latency overhead

At the main-table 32× setting, NanoKV adds less than 10% end-to-end latency relative to Original on each of the three backbones. From the rounded speedup values above, `(1 / speedup - 1) × 100` gives approximately 4.2% on HY-WorldPlay, 7.5% on LingBot-World 2.0, and 8.7% on Matrix-Game 3.5. This describes the standard NanoKV configuration; NanoKV-Fast timings are reported separately.

## NanoKV-Fast attention computation

The following values are arithmetic savings relative to NanoKV at nominal 32× compression, rather than measured latency reductions. The all-attention saving equals the historical share multiplied by the saving within history.

| Backbone | Historical share of visual attention | Historical-attention computation saved | All visual-attention computation saved |
|---|---:|---:|---:|
| HY-WorldPlay | 85.19% | 37.93% | 32.31% |
| Matrix-Game 3.5 | 72.00% | 20.83% | 15.00% |
| LingBot-World 2.0 | 71.43% | 34.32% | 24.51% |

For implementation and evaluation commands, see the [inference guide](../infer/README.md) and [training overview](../train/README.md).
