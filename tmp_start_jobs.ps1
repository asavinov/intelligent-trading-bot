$jobs = @()
$body = @{script_name='download_binance'} | ConvertTo-Json
for($i=0;$i -lt 3;$i++){
  $r = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/scripts/run' -Body $body -ContentType 'application/json'
  $jobs += $r.job_id
  Write-Output "Started job: $($r.job_id)"
  Start-Sleep -Seconds 1
}
$jobs | ConvertTo-Json -Compress
