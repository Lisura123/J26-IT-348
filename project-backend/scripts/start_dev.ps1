# start_dev.ps1 - Starts local development services for project-backend
# 1. Verifies .env variable names (without printing values)
# 2. Starts Mosquitto via project-backend docker-compose.yml
# 3. Launches MQTT ingest service and FastAPI server in separate windows
# 4. Records PIDs in .dev-pids and polls /health until ready

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "=== Starting Project Backend Dev Environment ===" -ForegroundColor Cyan

# 1. Check required .env variable names
$envFile = Join-Path $projectRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error ".env file not found at $envFile. Create it from .env.example."
    exit 1
}

$requiredKeys = @(
    "INFLUX_URL",
    "INFLUX_ORG",
    "INFLUX_BUCKET",
    "INFLUX_TOKEN",
    "MQTT_HOST",
    "MQTT_PORT",
    "API_KEY",
    "MEASUREMENT_PREFIX",
    "READ_API_KEY",
    "CORS_ORIGINS"
)

Write-Host "Verifying .env configuration keys..."
$foundKeys = @()
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line.Split("=", 2)
        if ($parts.Length -ge 1) {
            $foundKeys += $parts[0].Trim()
        }
    }
}

$missing = @()
foreach ($k in $requiredKeys) {
    if ($foundKeys -contains $k) {
        Write-Host "  [OK] Variable declared: $k" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] Variable missing: $k" -ForegroundColor Red
        $missing += $k
    }
}

if ($missing.Count -gt 0) {
    Write-Error "Missing required environment variables in .env: $($missing -join ', ')"
    exit 1
}

# 2. Start Mosquitto broker
Write-Host "`nStarting Mosquitto broker via Docker Compose..." -ForegroundColor Cyan
docker compose -f "$projectRoot\docker-compose.yml" up -d

# 3. Determine Python executable
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

# 4. Launch MQTT service and FastAPI API in separate windows
Write-Host "`nLaunching background services..." -ForegroundColor Cyan
$pids = @()

$mqttProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$pythonExe' -m backend.ingest.mqtt_service" -PassThru
$pids += $mqttProc.Id
Write-Host "  Started MQTT Service (PID: $($mqttProc.Id))"

$apiProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$pythonExe' -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000" -PassThru
$pids += $apiProc.Id
Write-Host "  Started FastAPI Server (PID: $($apiProc.Id))"

# Save PIDs to .dev-pids
$pidFile = Join-Path $projectRoot ".dev-pids"
$pids | Out-File -FilePath $pidFile -Encoding ascii
Write-Host "  Saved PIDs to $pidFile"

# 5. Wait for /health endpoint
Write-Host "`nWaiting for API health check..." -ForegroundColor Cyan
$healthUrl = "http://127.0.0.1:8000/health"
$maxAttempts = 45
$attempt = 0
$healthy = $false
$lastResp = $null

while ($attempt -lt $maxAttempts -and -not $healthy) {
    Start-Sleep -Seconds 1
    $attempt++
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2 -ErrorAction Stop
        $lastResp = $resp
        if ($resp.status -eq "ok" -and $resp.influx_connected -eq $true) {
            $healthy = $true
        }
    } catch {
        if ($_.Exception.Response) {
            try {
                $stream = $_.Exception.Response.GetResponseStream()
                $streamReader = New-Object System.IO.StreamReader($stream)
                $lastResp = $streamReader.ReadToEnd()
            } catch {}
        }
    }
}

if ($healthy) {
    Write-Host "  [OK] Backend API is healthy!" -ForegroundColor Green
} else {
    Write-Warning "API did not respond healthy within $maxAttempts seconds."
    if ($lastResp) {
        if ($lastResp -is [string]) {
            Write-Host "Health JSON: $lastResp" -ForegroundColor Yellow
        } else {
            $json = $lastResp | ConvertTo-Json -Depth 5
            Write-Host "Health JSON:`n$json" -ForegroundColor Yellow
        }
    } else {
        Write-Warning "API endpoint at $healthUrl could not be reached."
    }
}

# 6. Print connection endpoints
Write-Host "`n=== Development Services Ready ===" -ForegroundColor Green
Write-Host "  Health Status:    http://localhost:8000/health"
Write-Host "  Swagger UI Docs:  http://localhost:8000/docs"
Write-Host "  Query API:        http://localhost:8000/api/v1"
Write-Host "  Mosquitto Broker: localhost:1883"
Write-Host "`nTo stop all services started by this script, run .\scripts\stop_dev.ps1"
