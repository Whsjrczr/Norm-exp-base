# CFBN Focused W&B Summary

- generated: 2026-06-05 06:48 UTC
- nanoGPT project: `whsjrc-buaa/nanoGPT-CFBN-focused` (12 runs)
- ViT project: `whsjrc-buaa/ViT-CFBN-focused` (9 runs)
- all runs are seed 0 and finished unless noted in the tables.

## nanoGPT TinyShakespeare

| slot | norm | final val loss | best val loss | best ppl | final train loss | epoch | runtime(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | CCFBN | 1.5501 | 1.4698 | 4.3482 | 0.9918 | 30 | 705 |
| all | CCFBNc | 1.4934 | 1.4891 | 4.4330 | 1.3224 | 47 | 539 |
| all | CCFBNs | 1.6334 | 1.4525 | 4.2739 | 0.9403 | 18 | 667 |
| attn | CCFBN | 1.5397 | 1.4615 | 4.3126 | 0.9807 | 23 | 519 |
| attn | CCFBNc | 1.5077 | 1.4878 | 4.4274 | 1.1345 | 33 | 500 |
| attn | CCFBNs | 1.6003 | 1.4540 | 4.2803 | 0.9249 | 18 | 556 |
| final | CCFBN | 1.6015 | 1.4667 | 4.3350 | 0.9301 | 23 | 480 |
| final | CCFBNc | 1.5450 | 1.4763 | 4.3765 | 0.9915 | 28 | 469 |
| final | CCFBNs | 1.5441 | 1.4510 | 4.2676 | 0.9671 | 23 | 473 |
| mlp | CCFBN | 1.5815 | 1.4526 | 4.2741 | 0.9096 | 24 | 576 |
| mlp | CCFBNc | 1.4824 | 1.4702 | 4.3501 | 1.1621 | 47 | 498 |
| mlp | CCFBNs | 1.6417 | 1.4795 | 4.3907 | 0.9101 | 17 | 556 |

Best nanoGPT: `final/CCFBNs` with best val loss 1.4510 and perplexity 4.2676.

### nanoGPT by norm

| norm | mean best val loss | mean best ppl | best slot | best val loss |
|---|---:|---:|---|---:|
| CCFBN | 1.4626 | 4.3175 | mlp | 1.4526 |
| CCFBNc | 1.4808 | 4.3968 | mlp | 1.4702 |
| CCFBNs | 1.4593 | 4.3031 | final | 1.4510 |

## ViT CIFAR-10

| patch | norm | final test acc | best test acc | final train acc | gap | final test loss | epoch | runtime(s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | CFBN | 67.0300 | 67.1300 | 80.5469 | 13.5169 | 0.9479 | 189 | 12672 |
| 4 | CFBNc | 10.0000 | 20.5300 | 9.9980 | -0.0020 | 2.3026 | 0 | 8175 |
| 4 | CFBNs | 66.6400 | 66.7700 | 75.2724 | 8.6324 | 0.9511 | 185 | 8948 |
| 8 | CFBN | 60.7800 | 61.1600 | 82.4379 | 21.6579 | 1.1387 | 172 | 3187 |
| 8 | CFBNc | 18.1100 | 20.6300 | 17.9127 | -0.1973 | 2.1341 | 0 | 2798 |
| 8 | CFBNs | 60.6500 | 61.4900 | 83.1010 | 22.4510 | 1.1352 | 149 | 3052 |
| 16 | CFBN | 10.0000 | 28.7600 | 9.9920 | -0.0080 | 2.3026 | 2 | 1382 |
| 16 | CFBNc | 18.4800 | 19.2300 | 18.3514 | -0.1286 | 2.1140 | 2 | 1204 |
| 16 | CFBNs | 10.0000 | 25.0400 | 9.9920 | -0.0080 | 2.3026 | 2 | 1315 |

Best ViT: `patch4/CFBN` with best test acc 67.1300.

### ViT by patch

| patch | best norm | best test acc | CFBNc-CFBN delta | CFBNs-CFBN delta |
|---:|---|---:|---:|---:|
| 4 | CFBN | 67.1300 | -46.6000 | -0.3600 |
| 8 | CFBNs | 61.4900 | -40.5300 | 0.3300 |
| 16 | CFBN | 28.7600 | -9.5300 | -3.7200 |
