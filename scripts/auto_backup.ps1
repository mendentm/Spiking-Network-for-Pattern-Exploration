# auto_backup.ps1 — RSNN System
$ProjectDir  = "C:\Users\mende\Desktop\Projects\RSNN System"
$LogFile     = Join-Path $ProjectDir "logs\backup.log"
$CooldownSec = 3600
$DebounceMs  = 10000
$IgnoreDirs  = @(".git", ".venv", "__pycache__", "node_modules", "logs", ".pytest_cache")

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue
}

function Invoke-Backup {
    Set-Location $ProjectDir
    $status = git status --porcelain 2>&1
    if ($status) {
        git add -A 2>&1 | Out-Null
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
        git commit -m "chore: auto-backup $timestamp" --no-verify 2>&1 | Out-Null
        $remotes = git remote 2>&1
        if ($remotes -contains "origin") {
            git push origin main 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Log "Committed and pushed to origin/main" }
            else { Write-Log "Committed locally - push to origin failed" }
        } else { Write-Log "Committed locally (no remote configured)" }
    } else { Write-Log "Watcher triggered - no changes to commit" }
}

Write-Log "Watcher starting for: $ProjectDir"

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $ProjectDir
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, LastWrite, DirectoryName'
$watcher.EnableRaisingEvents = $true

$script:LastCommitTime = [DateTime]::MinValue
$script:DebounceTimer  = $null

function Start-DebounceTimer {
    if ($script:DebounceTimer) { $script:DebounceTimer.Stop(); $script:DebounceTimer.Dispose() }
    $script:DebounceTimer = New-Object System.Timers.Timer
    $script:DebounceTimer.Interval  = $DebounceMs
    $script:DebounceTimer.AutoReset = $false
    $elapsed = (New-TimeSpan -Start $script:LastCommitTime -End (Get-Date)).TotalSeconds
    $script:DebounceTimer.add_Elapsed({
        if ($elapsed -ge $CooldownSec) {
            Write-Log "Change detected - cooldown elapsed, running backup"
            Invoke-Backup
            $script:LastCommitTime = Get-Date
        } else {
            $remaining = [math]::Round($CooldownSec - $elapsed)
            Write-Log "Change detected - cooldown active, ${remaining}s remaining"
        }
    })
    $script:DebounceTimer.Start()
}

$action = {
    $path = $Event.SourceEventArgs.FullPath
    foreach ($d in $IgnoreDirs) { if ($path -like "*\$d\*") { return } }
    Start-DebounceTimer
}

Register-ObjectEvent $watcher "Changed" -Action $action | Out-Null
Register-ObjectEvent $watcher "Created" -Action $action | Out-Null
Register-ObjectEvent $watcher "Deleted" -Action $action | Out-Null
Register-ObjectEvent $watcher "Renamed" -Action $action | Out-Null

Write-Log "Watcher active. Commits at most once per hour on file change."
try { while ($true) { Start-Sleep -Seconds 30 } }
finally { $watcher.EnableRaisingEvents = $false; $watcher.Dispose(); Write-Log "Watcher stopped." }
