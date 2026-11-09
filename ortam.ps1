# Her yeni terminalde bir kez çalıştır:  .\ortam.ps1
$CatVTON = "C:\Users\Monster\Desktop\tryon"
& "$CatVTON\.venv\Scripts\Activate.ps1"
$env:CATVTON_DIR = $CatVTON
Set-Location $PSScriptRoot
Write-Host "Ortam hazir -> CatVTON: $CatVTON" -ForegroundColor Green