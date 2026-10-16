# stop_dev.ps1 - Stops development services started by start_dev.ps1
# Stops ONLY the PIDs listed in .dev-pids and their descendant processes via ParentProcessId.
# Plus 'docker compose stop' for this compose file if it started the container.
# NEVER selects processes by name. Never touches Windows services, other containers or volumes.

$ErrorActionPreference = "SilentlyContinue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "=== Stopping Project Backend Dev Services ===" -ForegroundColor Cyan

function Get-DescendantPids([int]$parentPid) {
    $descendants = @()
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $parentPid" -ErrorAction SilentlyContinue
    foreach ($c in $children) {
        $cPid = [int]$c.ProcessId
        $descendants += Get-DescendantPids $cPid
        $descendants += $cPid
    }
    return $descendants
}

# 1. Stop recorded process PIDs and their descendants
$pidFile = Join-Path $projectRoot ".dev-pids"
if (Test-Path $pidFile) {
    $rawPids = Get-Content $pidFile
    $pidsToStop = @()
    foreach ($line in $rawPids) {
        $p = [int]($line.Trim())
        if ($p -gt 0) {
            # Find all descendant processes first
            $children = Get-DescendantPids $p
            foreach ($cp in $children) {
                if ($pidsToStop -notcontains $cp) {
                    $pidsToStop += $cp
                }
            }
            if ($pidsToStop -notcontains $p) {
                $pidsToStop += $p
            }
        }
    }

    # Stop all collected PIDs (children first, then parents)
    foreach ($targetPid in $pidsToStop) {
        $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "  Stopping process PID $targetPid..."
            Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
        }
    }

    Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "  Removed $pidFile."
} else {
    Write-Host "  No .dev-pids file found."
}

# 2. If docker compose was started by start_dev.ps1, run 'docker compose stop' for this compose file
$dockerMarker = Join-Path $projectRoot ".dev-docker"
if (Test-Path $dockerMarker) {
    $composeFile = Join-Path $projectRoot "docker-compose.yml"
    Write-Host "  Stopping container started via docker compose..." -ForegroundColor Cyan
    docker compose -f "$composeFile" stop
    Remove-Item -Path $dockerMarker -Force -ErrorAction SilentlyContinue
    Write-Host "  Stopped docker-compose service and removed marker."
}

Write-Host "`n=== Dev Services Stopped Cleanly ===" -ForegroundColor Green
