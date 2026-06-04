param(
    [string]$InputFile = "",
    [string]$DataDir = "./dataset/text_translation",
    [string]$Output = "./results",
    [string]$SampleSrc = "hello",
    [int]$Gpu = 0
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$DefaultPython = Join-Path $RepoRoot ".conda\python.exe"
if (Test-Path $DefaultPython) {
    $PythonBin = $DefaultPython
} else {
    $PythonBin = "python"
}

$ArgsList = @(
    (Join-Path $RepoRoot "BERT\bert_translation.py"),
    "--data-dir", $DataDir,
    "--bert-layers", "4",
    "--bert-heads", "4",
    "--bert-embd", "256",
    "--max-src-len", "64",
    "--max-tgt-len", "64",
    "--batch-size", "64,64",
    "--iters-per-epoch", "100",
    "--eval-iters", "50",
    "--epochs", "50",
    "--optimizer", "adamw",
    "--lr", "3e-4",
    "--lr-method", "cos",
    "--weight-decay", "0.01",
    "--norm", "LN",
    "--activation", "gelu",
    "--dtype", "bfloat16",
    "--sample-src", $SampleSrc,
    "--gpu", "$Gpu",
    "--output", $Output
)

if ($InputFile -ne "") {
    $ArgsList += @("--input-file", $InputFile, "--overwrite-data")
}

& $PythonBin @ArgsList
