# auto_backup.ps1 — RSNN System
# Commits any uncommitted changes and pushes to remotes.
# Scheduled via Windows Task Scheduler (see scripts/setup_task.ps1 to register).

$ProjectDir = "C:\Users\mende\Desktop\Projects\RSNN System"
$LogFile    = Join-Path $ProjectDir "logs\backup.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Set-Location $ProjectDir

# Check for uncommitted changes (staged or unstaged)
$status = git status --porcelain
if ($status) {
    git add -A
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "chore: auto-backup $timestamp" --no-verify 2>&1 | Out-Null

    # Push to origin
    git push origin main 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Pushed to origin/main"
    } else {
        Write-Log "WARNING: Push to origin failed (no remote or auth issue)"
    }

    # Push to backup remote if configured
    $remotes = git remote
    if ($remotes -contains "backup") {
        git push backup main 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Pushed to backup remote"
        } else {
            Write-Log "WARNING: Push to backup remote failed"
        }
    }

    Write-Log "Auto-backup complete"
} else {
    Write-Log "No changes to commit"
}
