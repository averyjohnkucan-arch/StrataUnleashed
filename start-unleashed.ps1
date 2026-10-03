$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
foreach ($name in @('tmp', 'cache', 'config')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "work\$name") | Out-Null
}
$env:TEMP = Join-Path $PSScriptRoot 'work\tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:CUDA_CACHE_PATH = Join-Path $PSScriptRoot 'work\cache'
$env:XDG_CACHE_HOME = $env:CUDA_CACHE_PATH
$env:PIP_CACHE_DIR = $env:CUDA_CACHE_PATH
$env:XDG_CONFIG_HOME = Join-Path $PSScriptRoot 'work\config'
$env:PYTHONDONTWRITEBYTECODE = '1'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv }
    else { & python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Creating the local Python environment failed.' }
}
$receipt = Join-Path $PSScriptRoot '.venv\unleashed-dependencies-v1'
if (-not (Test-Path $receipt)) {
    & $python -B -m pip install --disable-pip-version-check -r requirements-unleashed.txt
    if ($LASTEXITCODE -ne 0) { throw 'Installing the local Python dependencies failed.' }
    New-Item -ItemType File -Force $receipt | Out-Null
}
if ($args.Count -eq 0 -or $args[0] -eq '--wizard') {
    $wizardArgs = @($args | Select-Object -Skip 1)
    & $python -B -m tools.unleashed_wizard @wizardArgs
} else {
    & $python -B unleashed.py @args
}
exit $LASTEXITCODE
