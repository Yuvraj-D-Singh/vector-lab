param([switch]$Production)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$server = Join-Path $root 'server'
$client = Join-Path $root 'client'

if (-not (Test-Path (Join-Path $server '.venv'))) {
  py -m venv (Join-Path $server '.venv')
}

& (Join-Path $server '.venv\\Scripts\\python.exe') -m pip install -r (Join-Path $server 'requirements.txt')

if ($Production) {
  Push-Location $client
  npm install
  npm run build
  Pop-Location
  & (Join-Path $server '.venv\\Scripts\\uvicorn.exe') app.main:app --app-dir $server --host 127.0.0.1 --port 8000
} else {
  Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$client'; npm install; npm run dev"
  & (Join-Path $server '.venv\\Scripts\\uvicorn.exe') app.main:app --app-dir $server --reload --port 8000
}
