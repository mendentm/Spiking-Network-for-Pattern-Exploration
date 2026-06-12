$TaskName = "ProjectBackup_rsnn-system"
$ScriptPath = "C:\Users\mende\Desktop\Projects\RSNN System\scripts\auto_backup.ps1"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}
$action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$ScriptPath`""
$trigger  = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 0) -StartWhenAvailable -DontStopOnIdleEnd -Priority 7
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Watcher backup for RSNN System" -RunLevel Limited | Out-Null
Write-Host "Registered: $TaskName"
Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
