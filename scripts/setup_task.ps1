# setup_task.ps1 — RSNN System
# Registers auto_backup.ps1 as a Windows Scheduled Task running every 30 minutes.
# Run this once per project: .\scripts\setup_task.ps1
# Requires: Run PowerShell as Administrator (or adjust trigger to current user only)

$ProjectName = "rsnn-system"
$ScriptPath  = "C:\Users\mende\Desktop\Projects\RSNN System\scripts\auto_backup.ps1"
$TaskName    = "ProjectBackup_$ProjectName"

# Remove existing task with the same name if re-running
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

$action  = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$ScriptPath`""

# Every 30 minutes, starting now
$trigger = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Minutes 30) -Once -At (Get-Date)

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable `
    -DontStopOnIdleEnd

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Auto-backup for $ProjectName every 30 minutes" `
    -RunLevel Limited | Out-Null

Write-Host "Scheduled task registered: $TaskName"
Write-Host "Runs every 30 minutes. To verify: Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "To remove:  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
