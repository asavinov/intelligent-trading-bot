param(
  [string[]] $JobIds,
  [int] $MaxAttempts = 300,    # number of polls per job (default 300 -> with 2s interval = 10 minutes)
  [int] $PollInterval = 2      # seconds between polls
)

# Normalize JobIds: support a single comma-separated string passed as one argument
if ($JobIds -and $JobIds.Count -eq 1 -and ($JobIds[0] -match ',')) {
  $JobIds = $JobIds[0].Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
}

# Ensure we operate from the repository root (script directory)
try {
  if ($PSScriptRoot) { Set-Location -Path $PSScriptRoot }
} catch { }

function Get-Job-Status($id){
  try{
    return Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/scripts/status/$id"
  } catch {
    # If API does not have the job in-memory (404), fall back to local metadata/log files
    try{
      $metaPath = Join-Path (Get-Location) "logs\jobs\$id.meta.json"
      if(Test-Path $metaPath){
        try{
          $metaRaw = Get-Content -Path $metaPath -Raw -ErrorAction Stop
          $meta = $metaRaw | ConvertFrom-Json
          # Support both server and ci_dummy keys for stdout path
          $stdoutPath = $null
          if ($meta.PSObject.Properties.Name -contains 'log_stdout' -and $meta.log_stdout) { $stdoutPath = $meta.log_stdout }
          elseif ($meta.PSObject.Properties.Name -contains 'stdout_path' -and $meta.stdout_path) { $stdoutPath = $meta.stdout_path }
          if ($stdoutPath) { $stdoutPath = $stdoutPath -replace '/', '\\' }
          if($stdoutPath -and -not (Test-Path $stdoutPath)){
            # Try relative path from repo root
            $stdoutPath = Join-Path (Get-Location) $stdoutPath
          }
          $stdout = ''
          if($stdoutPath -and (Test-Path $stdoutPath)){
            try{ $stdout = Get-Content -Path $stdoutPath -Raw -ErrorAction Stop } catch { $stdout = '' }
          }

          # Build a compatible object similar to /status response
          $rc = $null
          if ($meta.PSObject.Properties.Name -contains 'returncode') { $rc = $meta.returncode }
          if ($null -eq $rc -and ($meta.PSObject.Properties.Name -contains 'exit_code')) { $rc = $meta.exit_code }
          $st = $null
          if ($meta.PSObject.Properties.Name -contains 'status') { $st = $meta.status }
          if (-not $st) { if ($null -eq $rc -or $rc -eq 0 -or "$rc" -eq '0') { $st = 'completed' } else { $st = 'failed' } }
          $obj = [PSCustomObject]@{
            job_id = $meta.job_id
            status = $st
            returncode = $rc
            stdout = $stdout
          }
          return $obj
        } catch {
          return $null
        }
      }
    } catch {
      # ignore and return null so poller will retry per normal logic
    }
    return $null
  }
}

$results = @()
# Quick server health check: if the API is not reachable, prefer file-only mode and avoid long hangs
function Server-IsUp() {
  try {
    # Try a short GET to the scripts list endpoint with a small timeout
    $resp = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/scripts/list" -TimeoutSec 3 -ErrorAction Stop
    return $true
  } catch {
    return $false
  }
}

# If server is down or unreachable, do a file-only summary and exit quickly
if (-not (Server-IsUp)) {
  Write-Output "Server not reachable on http://127.0.0.1:8000 - performing file-only summary"
  foreach($id in $JobIds){
    $metaPath = Join-Path (Get-Location) "logs\jobs\$id.meta.json"
    if(Test-Path $metaPath){
      try{
        $metaRaw = Get-Content -Path $metaPath -Raw -ErrorAction Stop
        $meta = $metaRaw | ConvertFrom-Json
        # stdout path resolution for both schemas
        $stdoutPath = $null
        if ($meta.PSObject.Properties.Name -contains 'log_stdout' -and $meta.log_stdout) { $stdoutPath = $meta.log_stdout }
        elseif ($meta.PSObject.Properties.Name -contains 'stdout_path' -and $meta.stdout_path) { $stdoutPath = $meta.stdout_path }
        if ($stdoutPath) { $stdoutPath = $stdoutPath -replace '/', '\\' }
        if($stdoutPath -and -not (Test-Path $stdoutPath)){
          $stdoutPath = Join-Path (Get-Location) $stdoutPath
        }
        $stdout = ''
        if($stdoutPath -and (Test-Path $stdoutPath)){
          try{ $stdout = Get-Content -Path $stdoutPath -Raw -ErrorAction Stop } catch { $stdout = '' }
        }
        $snippet = ''
        if($stdout){
          $lines = $stdout -split "\n"
          $count = [math]::Min(5, $lines.Length)
          if($count -gt 0){ $snippet = $lines[0..($count-1)] -join "\n" }
        }
        $rc = $null
        if ($meta.PSObject.Properties.Name -contains 'returncode') { $rc = $meta.returncode }
        if ($null -eq $rc -and ($meta.PSObject.Properties.Name -contains 'exit_code')) { $rc = $meta.exit_code }
        $st = $null
        if ($meta.PSObject.Properties.Name -contains 'status') { $st = $meta.status }
        if (-not $st) { if ($null -eq $rc -or $rc -eq 0 -or "$rc" -eq '0') { $st = 'completed' } else { $st = 'failed' } }
        $entry = [ordered]@{
          job_id = $meta.job_id
          status = $st
          returncode = $rc
          stdout_snippet = $snippet
        }
        $results += $entry
      } catch {
        $results += [ordered]@{ job_id = $id; status = 'unknown'; returncode = $null; stdout_snippet = '' }
      }
    } else {
      $results += [ordered]@{ job_id = $id; status = 'not_found'; returncode = $null; stdout_snippet = '' }
    }
  }
  $outdir = "tools\ci_artifacts"
  if(-not (Test-Path $outdir)) { New-Item -ItemType Directory -Path $outdir | Out-Null }
  $json = $results | ConvertTo-Json -Depth 5
  $json | Out-File -Encoding utf8 (Join-Path $outdir 'ci_ui_runs_summary.json')
  Write-Output "Saved file-only summary to $outdir\ci_ui_runs_summary.json"
  if($results | Where-Object { $_.status -eq 'not_found' }){ exit 2 } else { exit 0 }
}
foreach($id in $JobIds){
  Write-Output "Polling job $id"
  $attempt = 0
  $timedOutThis = $false
  while($true){
    if($attempt -ge $MaxAttempts){
      Write-Output "Job $id timed out after $MaxAttempts attempts (poll interval ${PollInterval}s)."
      $entry = [ordered]@{
        job_id = $id
        status = 'timeout'
        returncode = $null
        stdout_snippet = ''
      }
      $results += $entry
      $timedOutThis = $true
      break
    }
    $attempt += 1
    $s = Get-Job-Status $id
    # If API didn't have the job (null), try an immediate local metadata fallback to avoid spamming the API with repeated requests
    if(-not $s){
      try{
        $metaPath = Join-Path (Get-Location) "logs\jobs\$id.meta.json"
        if(Test-Path $metaPath){
          $metaRaw = Get-Content -Path $metaPath -Raw -ErrorAction Stop
          $meta = $metaRaw | ConvertFrom-Json
          $stdoutPath = $null
          if ($meta.PSObject.Properties.Name -contains 'log_stdout' -and $meta.log_stdout) { $stdoutPath = $meta.log_stdout }
          elseif ($meta.PSObject.Properties.Name -contains 'stdout_path' -and $meta.stdout_path) { $stdoutPath = $meta.stdout_path }
          if ($stdoutPath) { $stdoutPath = $stdoutPath -replace '/', '\\' }
          if($stdoutPath -and -not (Test-Path $stdoutPath)){
            $stdoutPath = Join-Path (Get-Location) $stdoutPath
          }
          $stdout = ''
          if($stdoutPath -and (Test-Path $stdoutPath)){
            $stdout = Get-Content -Path $stdoutPath -Raw -ErrorAction SilentlyContinue
          }

          $rc = $null
          if ($meta.PSObject.Properties.Name -contains 'returncode') { $rc = $meta.returncode }
          if ($null -eq $rc -and ($meta.PSObject.Properties.Name -contains 'exit_code')) { $rc = $meta.exit_code }
          $st = $null
          if ($meta.PSObject.Properties.Name -contains 'status') { $st = $meta.status }
          if (-not $st) { if ($null -eq $rc -or $rc -eq 0 -or "$rc" -eq '0') { $st = 'completed' } else { $st = 'failed' } }

          $s = [PSCustomObject]@{
            job_id = $meta.job_id
            status = $st
            returncode = $rc
            stdout = $stdout
          }
        }
      } catch {
        # ignore and fall back to waiting
      }
    }
    if(-not $s){ Start-Sleep -Seconds $PollInterval; continue }
    if($s.status -eq 'running' -or $s.status -eq 'queued') { Start-Sleep -Seconds $PollInterval; continue }
    # finished or failed
    $snippet = ''
    if($s.stdout){
      $lines = $s.stdout -split "\n"
      $count = [math]::Min(5, $lines.Length)
      if($count -gt 0){ $snippet = $lines[0..($count-1)] -join "\n" }
    }
    $entry = [ordered]@{
      job_id = $s.job_id
      status = $s.status
      returncode = $s.returncode
      stdout_snippet = $snippet
    }
    $results += $entry
    break
  }
  if($timedOutThis){ $global:AnyTimedOut = $true }
}

$outdir = "tools\ci_artifacts"
if(-not (Test-Path $outdir)) { New-Item -ItemType Directory -Path $outdir | Out-Null }
$json = $results | ConvertTo-Json -Depth 5
$json | Out-File -Encoding utf8 (Join-Path $outdir 'ci_ui_runs_summary.json')
Write-Output "Saved summary to $outdir\ci_ui_runs_summary.json"
if($global:AnyTimedOut){
  Write-Output "One or more jobs timed out. Exiting with code 2."
  exit 2
} else {
  exit 0
}
