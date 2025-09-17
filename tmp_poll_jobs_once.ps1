param(
  [string[]] $JobIds
)

# Single-pass poller: for each job, try API once, if missing read local metadata and stdout, then write summary JSON.

# Normalize JobIds: support a single comma-separated string passed as one argument
if ($JobIds -and $JobIds.Count -eq 1 -and ($JobIds[0] -match ',')) {
  $JobIds = $JobIds[0].Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
}

# Ensure we operate from the repository root (script directory)
try {
  if ($PSScriptRoot) { Set-Location -Path $PSScriptRoot }
} catch { }

$outdir = "tools\ci_artifacts"
if(-not (Test-Path $outdir)) { New-Item -ItemType Directory -Path $outdir | Out-Null }
$results = @()
foreach($id in $JobIds){
  Write-Output "Checking job $id"
  $obj = $null
  try{
    $resp = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/scripts/status/$id" -ErrorAction Stop
    $snippet = ''
    if ($resp -and $resp.stdout) {
      $lines = ($resp.stdout -split "\n")
      if ($lines -and $lines.Length -gt 0) {
        $count = [math]::Min(5, $lines.Length)
        $snippet = $lines[0..($count-1)] -join "\n"
      }
    }
    $obj = [PSCustomObject]@{
      job_id = $resp.job_id
      status = $resp.status
      returncode = $resp.returncode
      stdout_snippet = $snippet
    }
  } catch {
    # fallback to local metadata
    $metaPath = Join-Path (Get-Location) "logs\jobs\$id.meta.json"
    if(Test-Path $metaPath){
      try{
        $metaRaw = Get-Content -Path $metaPath -Raw -ErrorAction Stop
        $meta = $metaRaw | ConvertFrom-Json
        # Support both keys: log_stdout (server) or stdout_path (ci_dummy)
        $stdoutPath = $null
        if ($meta.PSObject.Properties.Name -contains 'log_stdout' -and $meta.log_stdout) {
          $stdoutPath = $meta.log_stdout
        } elseif ($meta.PSObject.Properties.Name -contains 'stdout_path' -and $meta.stdout_path) {
          $stdoutPath = $meta.stdout_path
        }
        if ($stdoutPath) { $stdoutPath = $stdoutPath -replace '/', '\\' }
        if ($stdoutPath -and -not (Test-Path $stdoutPath)) {
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
        # Normalize return code and status
        $rc = $null
        if ($meta.PSObject.Properties.Name -contains 'returncode') { $rc = $meta.returncode }
        if ($null -eq $rc -and ($meta.PSObject.Properties.Name -contains 'exit_code')) { $rc = $meta.exit_code }
        $st = $null
        if ($meta.PSObject.Properties.Name -contains 'status') { $st = $meta.status }
        if (-not $st) {
          if ($null -eq $rc -or $rc -eq 0 -or "$rc" -eq '0') { $st = 'completed' } else { $st = 'failed' }
        }
        $obj = [PSCustomObject]@{
          job_id = $meta.job_id
          status = $st
          returncode = $rc
          stdout_snippet = $snippet
        }
      } catch {
        $obj = [PSCustomObject]@{ job_id = $id; status = 'unknown'; returncode = $null; stdout_snippet = '' }
      }
    } else {
      $obj = [PSCustomObject]@{ job_id = $id; status = 'not_found'; returncode = $null; stdout_snippet = '' }
    }
  }
  $results += $obj
}

$results | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $outdir 'ci_ui_runs_summary.json')
Write-Output "Wrote summary to $outdir\ci_ui_runs_summary.json"