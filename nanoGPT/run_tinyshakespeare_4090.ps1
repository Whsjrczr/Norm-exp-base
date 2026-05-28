python nanoGPT/prepare_tinyshakespeare.py --data-dir ./dataset/tinyshakespeare

python nanoGPT/nanogpt.py `
  --data-dir ./dataset/tinyshakespeare `
  --batch-size 128,128 `
  --iters-per-epoch 100 `
  --eval-iters 50 `
  --epochs 50 `
  --optimizer adamw `
  --lr 6e-4 `
  --lr-method cos `
  --weight-decay 0.1 `
  --norm LN `
  --activation gelu `
  --dtype bfloat16 `
  --compile
