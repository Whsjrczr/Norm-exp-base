# CFBN vs SBN/SeqBN W&B Comparison

- generated: 2026-06-04 08:11 UTC
- projects: `whsjrc-buaa/nanoGPT-CFBN-focused`, `whsjrc-buaa/nanoGPT-SeqBN`, `whsjrc-buaa/ViT-CFBN-focused`, `whsjrc-buaa/ViT-SeqBN-patch8`, `whsjrc-buaa/ViT-SeqBN-seed-validation`, `whsjrc-buaa/ViT-SeqBN-failure-diagnostics`

## nanoGPT, lr=6e-4, seed=0

| family | norm | slot | best val loss | best ppl | final val loss |
|---|---|---|---:|---:|---:|
| CFBN | CCFBN | all | 1.5501 | 4.7120 | 1.5501 |
| CFBN | CCFBN | attn | 1.5397 | 4.6630 | 1.5397 |
| CFBN | CCFBN | final | 1.6015 | 4.9607 | 1.6015 |
| CFBN | CCFBN | mlp | 1.5815 | 4.8624 | 1.5815 |
| CFBN | CCFBNc | all | 1.4934 | 4.4520 | 1.4934 |
| CFBN | CCFBNc | attn | 1.5077 | 4.5165 | 1.5077 |
| CFBN | CCFBNc | final | 1.5450 | 4.6881 | 1.5450 |
| CFBN | CCFBNc | mlp | 1.4824 | 4.4037 | 1.4824 |
| CFBN | CCFBNs | all | 1.6334 | 5.1214 | 1.6334 |
| CFBN | CCFBNs | attn | 1.6003 | 4.9547 | 1.6003 |
| CFBN | CCFBNs | final | 1.5441 | 4.6837 | 1.5441 |
| CFBN | CCFBNs | mlp | 1.6417 | 5.1642 | 1.6417 |
| SBN/SeqBN | CDSeqBN | all | 1.5726 | 4.8190 | 1.5726 |
| SBN/SeqBN | CDSeqBN | attn | 1.5869 | 4.8888 | 1.5869 |
| SBN/SeqBN | CDSeqBN | final | 1.5719 | 4.8159 | 1.5719 |
| SBN/SeqBN | CDSeqBN | mlp | 1.5553 | 4.7366 | 1.5553 |
| SBN/SeqBN | CDSeqBNc | all | 1.4891 | 4.4330 | 1.4891 |
| SBN/SeqBN | CDSeqBNc | mlp | 1.4746 | 4.3691 | 1.4746 |
| SBN/SeqBN | CDSeqBNs | all | 1.5649 | 4.7820 | 1.5649 |
| SBN/SeqBN | CDSeqBNs | attn | 1.5719 | 4.8160 | 1.5719 |
| SBN/SeqBN | CDSeqBNs | final | 1.5732 | 4.8223 | 1.5732 |
| SBN/SeqBN | CDSeqBNs | mlp | 1.5717 | 4.8148 | 1.5717 |
| SBN/SeqBN | CSBN | all | 1.5537 | 4.7289 | 1.5537 |
| SBN/SeqBN | CSBN | attn | 1.5087 | 4.5207 | 1.5087 |
| SBN/SeqBN | CSBN | final | 1.5385 | 4.6576 | 1.5385 |
| SBN/SeqBN | CSBN | mlp | 1.4626 | 4.3172 | 1.4626 |
| SBN/SeqBN | CSBNc | all | 1.5218 | 4.5803 | 1.5218 |
| SBN/SeqBN | CSBNc | attn | 1.5134 | 4.5421 | 1.5134 |
| SBN/SeqBN | CSBNc | final | 1.5429 | 4.6782 | 1.5429 |
| SBN/SeqBN | CSBNc | mlp | 1.4939 | 4.4545 | 1.4939 |
| SBN/SeqBN | CSBNs | final | 17.0414 | 25176191.4165 | 17.0414 |
| SBN/SeqBN | CSBNs | mlp | 2.6034 | 13.5101 | 2.6034 |
| SBN/SeqBN | CSeqBN | all | 1.5948 | 4.9274 | 1.5948 |
| SBN/SeqBN | CSeqBN | attn | 1.5797 | 4.8536 | 1.5797 |
| SBN/SeqBN | CSeqBN | final | 1.5799 | 4.8545 | 1.5799 |
| SBN/SeqBN | CSeqBN | mlp | 1.5614 | 4.7656 | 1.5614 |
| SBN/SeqBN | CSeqBNc | all | 1.4863 | 4.4207 | 1.4863 |
| SBN/SeqBN | CSeqBNc | mlp | 1.4984 | 4.4747 | 1.4984 |
| SBN/SeqBN | CSeqBNs | all | 1.5971 | 4.9387 | 1.5971 |
| SBN/SeqBN | CSeqBNs | attn | 1.5820 | 4.8649 | 1.5820 |
| SBN/SeqBN | CSeqBNs | final | 1.5761 | 4.8360 | 1.5761 |
| SBN/SeqBN | CSeqBNs | mlp | 1.5768 | 4.8396 | 1.5768 |
| SBN/SeqBN | LN | all | 1.5855 | 4.8818 | 1.5855 |
| SBN/SeqBN | RMS | all | 1.5799 | 4.8546 | 1.5799 |
| SBN/SeqBN | RMS | attn | 1.5718 | 4.8154 | 1.5718 |
| SBN/SeqBN | RMS | final | 1.5855 | 4.8817 | 1.5855 |
| SBN/SeqBN | RMS | mlp | 1.5787 | 4.8486 | 1.5787 |

### nanoGPT slot/family summary

| family | norm | mean best val loss | n | best slot | best val loss |
|---|---|---:|---:|---|---:|
| CFBN | CCFBN | 1.5682 | 4 | attn | 1.5397 |
| CFBN | CCFBNc | 1.5071 | 4 | mlp | 1.4824 |
| CFBN | CCFBNs | 1.6049 | 4 | final | 1.5441 |
| SBN/SeqBN | CDSeqBN | 1.5717 | 4 | mlp | 1.5553 |
| SBN/SeqBN | CDSeqBNc | 1.4818 | 2 | mlp | 1.4746 |
| SBN/SeqBN | CDSeqBNs | 1.5704 | 4 | all | 1.5649 |
| SBN/SeqBN | CSBN | 1.5159 | 4 | mlp | 1.4626 |
| SBN/SeqBN | CSBNc | 1.5180 | 4 | mlp | 1.4939 |
| SBN/SeqBN | CSBNs | 9.8224 | 2 | mlp | 2.6034 |
| SBN/SeqBN | CSeqBN | 1.5790 | 4 | mlp | 1.5614 |
| SBN/SeqBN | CSeqBNc | 1.4924 | 2 | all | 1.4863 |
| SBN/SeqBN | CSeqBNs | 1.5830 | 4 | final | 1.5761 |
| SBN/SeqBN | LN | 1.5855 | 1 | all | 1.5855 |
| SBN/SeqBN | RMS | 1.5790 | 4 | attn | 1.5718 |

## ViT comparable slices

| slice | norm | source | mean best test acc | n | final test acc mean |
|---|---|---|---:|---:|---:|
| patch4 lr1e-4 | CFBN | cfbn_vit | 67.03 | 1 | 67.03 |
| patch4 lr1e-4 | CFBNc | cfbn_vit | 10.00 | 1 | 10.00 |
| patch4 lr1e-4 | CFBNs | cfbn_vit | 66.64 | 1 | 66.64 |
| patch4 lr1e-4 | DSeqBN | vit_seed | 62.37 | 3 | 62.37 |
| patch4 lr1e-4 | DSeqBNs | vit_seed | 58.48 | 3 | 58.48 |
| patch4 lr1e-4 | LN | vit_seed | 55.17 | 3 | 55.17 |
| patch4 lr1e-4 | SBN | vit_seed | 67.79 | 3 | 67.79 |
| patch4 lr1e-4 | SeqBN | vit_seed | 57.90 | 3 | 57.90 |
| patch4 lr1e-4 | SeqBNs | vit_seed | 58.04 | 3 | 58.04 |
| patch8 lr1e-4 | CFBN | cfbn_vit | 60.78 | 1 | 60.78 |
| patch8 lr1e-4 | CFBNc | cfbn_vit | 18.11 | 1 | 18.11 |
| patch8 lr1e-4 | CFBNs | cfbn_vit | 60.65 | 1 | 60.65 |
| patch8 lr1e-4 | DSeqBN | vit_patch8 | 60.69 | 2 | 60.69 |
| patch8 lr1e-4 | LN | vit_patch8 | 55.06 | 2 | 55.06 |
| patch8 lr1e-4 | SBN | vit_patch8 | 59.68 | 2 | 59.68 |
| patch8 lr1e-4 | SBNs | vit_diag | 59.31 | 3 | 59.31 |
| patch8 lr1e-4 | SeqBN | vit_patch8 | 57.75 | 2 | 57.75 |
| patch16 lr1e-3 | CFBN | cfbn_vit | 10.00 | 1 | 10.00 |
| patch16 lr1e-3 | CFBNc | cfbn_vit | 18.48 | 1 | 18.48 |
| patch16 lr1e-3 | CFBNs | cfbn_vit | 10.00 | 1 | 10.00 |
| patch16 lr1e-3 | DSeqBN | vit_seed | 51.26 | 3 | 51.26 |
| patch16 lr1e-3 | DSeqBNs | vit_seed | 52.68 | 3 | 52.68 |
| patch16 lr1e-3 | LN | vit_seed | 37.16 | 3 | 37.16 |
| patch16 lr1e-3 | SBN | vit_diag,vit_seed | 21.00 | 4 | 21.00 |
| patch16 lr1e-3 | SBNs | vit_diag | 10.00 | 4 | 10.00 |
| patch16 lr1e-3 | SeqBN | vit_seed | 47.12 | 3 | 47.12 |
| patch16 lr1e-3 | SeqBNs | vit_seed | 45.27 | 3 | 45.27 |
| patch16 lr1e-4 | DSeqBN | vit_seed | 52.89 | 3 | 52.89 |
| patch16 lr1e-4 | DSeqBNs | vit_seed | 52.76 | 3 | 52.76 |
| patch16 lr1e-4 | LN | vit_seed | 50.62 | 3 | 50.62 |
| patch16 lr1e-4 | SBN | vit_diag,vit_seed | 49.29 | 4 | 49.29 |
| patch16 lr1e-4 | SBNs | vit_diag | 10.00 | 4 | 10.00 |
| patch16 lr1e-4 | SeqBN | vit_seed | 50.84 | 3 | 50.84 |
| patch16 lr1e-4 | SeqBNs | vit_seed | 50.51 | 3 | 50.51 |
