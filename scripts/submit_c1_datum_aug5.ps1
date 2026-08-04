param(
    [string]$NotBefore = "2026-08-05T08:05:00+08:00"
)

$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\envs\kaggle-dev\python.exe"
$Competition = "rogii-wellbore-geology-prediction"
$LogDirectory = Join-Path $Repo "kaggle\submission_logs"
$LogPath = Join-Path $LogDirectory "c1_datum_aug5.log"
$StatePath = Join-Path $LogDirectory "c1_datum_aug5.completed"
$StartAt = [DateTimeOffset]::Parse($NotBefore)

# Final-day portfolio: every slot is a directly competitive candidate.
$Jobs = @(
    @{
        Slug = "rogii-c1r8-d16-b12-calibrated"
        Version = 1
        Output = "kaggle\outputs\rogii-c1r8-d16-b12-calibrated-v1"
        Message = "joint rank8 datum plus b1 b2 contrast calibration"
    },
    @{
        Slug = "rogii-c1r8-d16-b12-cap4"
        Version = 1
        Output = "kaggle\outputs\rogii-c1r8-d16-b12-cap4-v1"
        Message = "aggressive b12 calibrated datum offset cap4"
    },
    @{
        Slug = "rogii-c1r8-d16-b12-a0b3-probe"
        Version = 1
        Output = "kaggle\outputs\rogii-c1r8-d16-b12-a0b3-probe-v1"
        Message = "competitive b12 plus b3 direction a0 amplitude1"
    },
    @{
        Slug = "rogii-c1r8-d16-b12-a1b3-probe"
        Version = 1
        Output = "kaggle\outputs\rogii-c1r8-d16-b12-a1b3-probe-v1"
        Message = "competitive b12 plus b3 direction a1 amplitude1"
    },
    @{
        Slug = "rogii-c1r8-d16-b12-a2b3-probe"
        Version = 1
        Output = "kaggle\outputs\rogii-c1r8-d16-b12-a2b3-probe-v1"
        Message = "competitive b12 plus b3 direction a2 amplitude1"
    }
)

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f ([DateTimeOffset]::Now.ToString("o")), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

if (Test-Path -LiteralPath $StatePath) {
    Write-Log "completion marker already exists; exiting without duplicate submissions"
    exit 0
}

foreach ($Job in $Jobs) {
    $submission = Join-Path (Join-Path $Repo $Job.Output) "submission.csv"
    if (-not (Test-Path -LiteralPath $submission)) {
        throw "required audited submission is missing: $submission"
    }
}

Write-Log "worker ready; not-before=$($StartAt.ToString('o')) jobs=$($Jobs.Count) all-direct-candidates=true"
while ([DateTimeOffset]::Now -lt $StartAt) {
    $remaining = ($StartAt - [DateTimeOffset]::Now).TotalSeconds
    Start-Sleep -Seconds ([Math]::Max(1, [Math]::Min(300, [Math]::Ceiling($remaining))))
}

foreach ($Job in $Jobs) {
    $outputDirectory = Join-Path $Repo $Job.Output
    $accepted = $false
    for ($attempt = 1; $attempt -le 36; $attempt++) {
        Write-Log "submit start slug=$($Job.Slug) version=$($Job.Version) attempt=$attempt"
        Push-Location $outputDirectory
        try {
            $response = & $Python -m kaggle competitions submit `
                -c $Competition `
                -f submission.csv `
                -k "qwer556617123/$($Job.Slug)" `
                -v $Job.Version `
                -m $Job.Message 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            Pop-Location
        }
        foreach ($line in $response) {
            Write-Log "kaggle: $line"
        }
        if ($exitCode -eq 0) {
            Write-Log "submit accepted slug=$($Job.Slug) version=$($Job.Version)"
            $accepted = $true
            break
        }
        Write-Log "submit failed slug=$($Job.Slug) exit=$exitCode; retrying in 600 seconds"
        Start-Sleep -Seconds 600
    }
    if (-not $accepted) {
        Write-Log "giving up slug=$($Job.Slug) after 36 attempts"
        exit 1
    }
}

Set-Content -LiteralPath $StatePath -Value ([DateTimeOffset]::Now.ToString("o")) -Encoding UTF8
Write-Log "all five direct candidate submissions accepted"
