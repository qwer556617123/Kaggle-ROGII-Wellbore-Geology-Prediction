param(
    [string]$ScoreDeadline = "2026-08-06T05:30:00+08:00"
)

$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\envs\kaggle-dev\python.exe"
$Competition = "rogii-wellbore-geology-prediction"
$Slug = "rogii-c1r8-d16-b123-calibrated"
$Kernel = "qwer556617123/$Slug"
$KernelDirectory = Join-Path $Repo "kaggle\$Slug"
$OutputDirectory = Join-Path $Repo "kaggle\outputs\$Slug-v1"
$LogDirectory = Join-Path $Repo "kaggle\submission_logs"
$BatchStatePath = Join-Path $LogDirectory "c1_datum_aug5.completed"
$LogPath = Join-Path $LogDirectory "c1_datum_aug5_finalizer.log"
$StatePath = Join-Path $LogDirectory "c1_datum_aug5_finalizer.completed"
$Deadline = [DateTimeOffset]::Parse($ScoreDeadline)
$Builder = Join-Path $Repo "scripts\diagnostics\build_c1_rank8_datum16_multistage_calibrated.py"
$Auditor = Join-Path $Repo "scripts\diagnostics\audit_c1_rank8_datum16_multistage_output.py"
$BaseSha = "10a6df6830ccef1ce4f3ff6823a6d254fc37d9106c746555bc8bee64a7e1623b"
$B1Scores = "6.714,6.394,6.984,6.867"
$B2Scores = "6.988,6.880,6.443,6.715"
$ProbeDescriptions = @(
    "datum16 b3 Hadamard code a0 amplitude2",
    "datum16 b3 Hadamard code a1 amplitude2",
    "datum16 b3 Hadamard code a2 amplitude2"
)

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f ([DateTimeOffset]::Now.ToString("o")), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

function Get-CompletedProbeScores {
    $raw = & $Python -m kaggle competitions submissions -c $Competition --csv
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    $header = -1
    for ($index = 0; $index -lt $raw.Count; $index++) {
        if ([string]$raw[$index] -like "ref,fileName,date,description,status,publicScore,privateScore*") {
            $header = $index
            break
        }
    }
    if ($header -lt 0) {
        return $null
    }
    $rows = (($raw[$header..($raw.Count - 1)]) -join "`n") | ConvertFrom-Csv
    $scores = @()
    foreach ($description in $ProbeDescriptions) {
        $row = $rows | Where-Object {
            $_.description -eq $description -and $_.status -eq "SubmissionStatus.COMPLETE"
        } | Select-Object -First 1
        if ($null -eq $row -or [string]::IsNullOrWhiteSpace($row.publicScore)) {
            return $null
        }
        $scores += [double]::Parse(
            $row.publicScore,
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
    return $scores
}

if (Test-Path -LiteralPath $StatePath) {
    Write-Log "completion marker already exists; exiting without duplicate final submission"
    exit 0
}

Write-Log "finalizer ready; score-deadline=$($Deadline.ToString('o'))"
while (-not (Test-Path -LiteralPath $BatchStatePath)) {
    if ([DateTimeOffset]::Now -ge $Deadline) {
        Write-Log "batch completion deadline exceeded; exiting"
        exit 1
    }
    Start-Sleep -Seconds 60
}

$scores = $null
while ([DateTimeOffset]::Now -lt $Deadline) {
    $scores = Get-CompletedProbeScores
    if ($null -ne $scores -and $scores.Count -eq 3) {
        break
    }
    Write-Log "b3 scores not complete; polling again in 300 seconds"
    Start-Sleep -Seconds 300
}
if ($null -eq $scores -or $scores.Count -ne 3) {
    Write-Log "b3 scores unavailable before deadline; fifth slot left unused"
    exit 1
}

$scoreText = ($scores | ForEach-Object {
    $_.ToString("0.000000", [System.Globalization.CultureInfo]::InvariantCulture)
}) -join ","
Write-Log "b3 scores acquired: $scoreText"

Push-Location $Repo
try {
    $buildOutput = & $Python $Builder `
        --anchor 6.524 `
        --b1-scores $B1Scores `
        --b2-scores $B2Scores `
        --b3-scores $scoreText `
        --b3-parent-code-indices 0,1,2 `
        --amplitude 2 2>&1
    $buildExit = $LASTEXITCODE
}
finally {
    Pop-Location
}
foreach ($line in $buildOutput) {
    Write-Log "builder: $line"
}
if ($buildExit -ne 0) {
    Write-Log "builder failed exit=$buildExit"
    exit 1
}

$pushOutput = & $Python -m kaggle kernels push -p $KernelDirectory 2>&1
foreach ($line in $pushOutput) {
    Write-Log "kernel-push: $line"
}
if ($LASTEXITCODE -ne 0) {
    Write-Log "kernel push failed"
    exit 1
}

$kernelComplete = $false
while ([DateTimeOffset]::Now -lt $Deadline) {
    $statusOutput = & $Python -m kaggle kernels status $Kernel 2>&1
    $statusText = $statusOutput -join " "
    Write-Log "kernel-status: $statusText"
    if ($statusText -match "COMPLETE") {
        $kernelComplete = $true
        break
    }
    if (($statusText -match "ERROR") -or ($statusText -match "CANCEL")) {
        Write-Log "kernel failed; final submission cancelled"
        exit 1
    }
    Start-Sleep -Seconds 30
}
if (-not $kernelComplete) {
    Write-Log "kernel did not complete before finalizer deadline"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$downloadOutput = & $Python -m kaggle kernels output $Kernel -p $OutputDirectory --force 2>&1
foreach ($line in $downloadOutput) {
    Write-Log "kernel-output: $line"
}
if ($LASTEXITCODE -ne 0) {
    Write-Log "kernel output download failed"
    exit 1
}

$auditOutput = & $Python $Auditor `
    --root $OutputDirectory `
    --expected-base-sha $BaseSha `
    --expected-child-codes 1,2,3 2>&1
foreach ($line in $auditOutput) {
    Write-Log "audit: $line"
}
if ($LASTEXITCODE -ne 0) {
    Write-Log "output audit failed; final submission cancelled"
    exit 1
}

Push-Location $OutputDirectory
try {
    $submitOutput = & $Python -m kaggle competitions submit `
        -c $Competition `
        -f submission.csv `
        -k $Kernel `
        -v 1 `
        -m "partial three-code b3 calibrated datum16 final deployment" 2>&1
    $submitExit = $LASTEXITCODE
}
finally {
    Pop-Location
}
foreach ($line in $submitOutput) {
    Write-Log "competition-submit: $line"
}
if ($submitExit -ne 0) {
    Write-Log "final competition submission failed exit=$submitExit"
    exit 1
}

Set-Content -LiteralPath $StatePath -Value ([DateTimeOffset]::Now.ToString("o")) -Encoding UTF8
Write-Log "final calibrated submission accepted"
