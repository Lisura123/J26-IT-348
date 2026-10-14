# stop_dev.ps1 - Stops development services started by start_dev.ps1
# 1. Terminates PowerShell background worker processes recorded in .dev-pids
# 2. Stops Mosquitto container via project-backend docker-compose.yml ONLY
# Never stops, inspects, or touches any other containers.

$ErrorActionPreference = "SilentlyContinue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "=== Stopping Project Backend Dev Services ===" -ForegroundColor Cyan

# 1. Stop recorded process PIDs
$pidFile = Join-Path $projectRoot ".dev-pids"
if (Test-Path $pidFile) {
    $pids = Get-Content $pidFile
    foreach ($p in $pids) {
        $procId = [int]$p.Trim()
        if ($procId -gt 0) {
            $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "  Stopping process $procId ($($proc.ProcessName))..."
                # Stop process tree (powershell window and child python processes)
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            } else {
                Write-Host "  Process $procId is already stopped."
            }
        }
    }
    Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "  Removed $pidFile."
} else {
    Write-Host "  No .dev-pids file found."
}

# Also ensure any lingering python processes running uvicorn or mqtt_service from this repo are stopped
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
        if ($cmd -match "backend\.ingest\.mqtt_service" -or $cmd -match "backend\.api\.main:app") {
            Write-Host "  Stopping lingering worker PID $($_.Id)..."
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }
    } catch {}
}

# 2. Stop Mosquitto container for this project only
Write-Host "`nStopping Mosquitto container via Docker Compose..." -ForegroundColor Cyan
docker compose -f "$projectRoot\docker-compose.yml" stop

Write-Host "`n=== Dev Services Stopped Cleanly ===" -ForegroundColor Green
