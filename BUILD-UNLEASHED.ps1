param([string]$CudaArch = 'native', [int]$Jobs = 8)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path 'work\tmp', 'work\cache' | Out-Null
$env:TEMP = Join-Path $PSScriptRoot 'work\tmp'
$env:TMP = $env:TEMP
$env:CUDA_CACHE_PATH = Join-Path $PSScriptRoot 'work\cache'
foreach ($tool in @('cmake', 'ninja', 'cl', 'nvcc')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing $tool. Use an x64 Visual Studio developer terminal with CUDA Toolkit, CMake and Ninja on PATH."
    }
}
$ggml = (Join-Path $PSScriptRoot 'third_party\llama.cpp').Replace('\', '/')
& cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON `
    -DSTRATA_PORTABLE=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" "-DSTRATA_GGML_DIR=$ggml"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& cmake --build build --target strata native_expert_parity kv_mixed_parity ple_q8_parity --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Engine build failed.' }
Write-Host 'Built build\strata.exe. Run START-UNLEASHED.bat --model C:\path\model.gguf'
