$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$DefaultPython = Join-Path $RepoRoot ".conda\python.exe"
if (Test-Path $DefaultPython) {
    $PythonBin = $DefaultPython
} else {
    $PythonBin = "python"
}

& $PythonBin (Join-Path $RepoRoot "BERT\bert_translation.py") `
  --bert-layers 1 `
  --bert-heads 2 `
  --bert-embd 32 `
  --max-src-len 8 `
  --max-tgt-len 8 `
  --batch-size 2,2 `
  --iters-per-epoch 1 `
  --eval-iters 1 `
  --epochs 1 `
  --dtype float32 `
  --no-save-checkpoint `
  --sample-src "hello"
