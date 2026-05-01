param(
  [switch]$All = $false
)

$ErrorActionPreference = "Stop"

function Remove-IfExists($Path) {
  if (Test-Path $Path) {
    Remove-Item -Force -Recurse -LiteralPath $Path
  }
}

# Python cache/artifacts
Get-ChildItem -Recurse -Force -Directory -Filter "__pycache__" | ForEach-Object { Remove-IfExists $_.FullName }
Get-ChildItem -Recurse -Force -File -Include "*.pyc","*.pyo" | ForEach-Object { Remove-Item -Force -LiteralPath $_.FullName }

# Frontend build output
Remove-IfExists "frontend\\dist"

if ($All) {
  # Optional heavy cleanup
  Remove-IfExists "frontend\\node_modules"
  Remove-IfExists "node_modules"
}

Write-Host "Clean complete."

