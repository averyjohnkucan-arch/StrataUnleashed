# Run from PowerShell. Installs Python if needed, then opens model selection and download.
$ErrorActionPreference = 'Stop'
$installDir = if ($env:STRATA_INSTALL_DIR) { $env:STRATA_INSTALL_DIR } else { Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'StrataUnleashed' }
New-Item -ItemType Directory -Force (Join-Path $installDir 'work\installer'), (Join-Path $installDir 'work\tmp') | Out-Null
$env:TEMP = Join-Path $installDir 'work\tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
function Find-StrataPython {
    $candidates = @((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'))
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $found = & py -3 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $candidates += $found }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += (Get-Command python).Source }
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            & $candidate -c 'import sys; assert sys.version_info >= (3,10)' 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    return $null
}
$python = Find-StrataPython
if (-not $python) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw 'Install Python 3.10+ from python.org, then run this command again.' }
    Write-Host 'Installing Python for your user account ...'
    & winget install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Python installation did not finish. Install Python and rerun setup.' }
    $python = Find-StrataPython
    if (-not $python) { throw 'Python was installed; open a new PowerShell window and rerun setup.' }
}
$env:STRATA_PYTHON = $python
$script = Join-Path $installDir 'work\installer\install_unleashed.py'
Invoke-WebRequest -UseBasicParsing 'https://raw.githubusercontent.com/averyjohnkucan-arch/StrataUnleashed/unleashed/install_unleashed.py' -OutFile $script
& $python -B $script --directory $installDir @args
if ($LASTEXITCODE -ne 0) { throw 'Strata setup stopped. See the message above.' }
