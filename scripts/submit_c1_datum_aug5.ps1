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

# Deliberately submit four jobs. The fifth daily slot is reserved for the
# partial-b3 calibrated deployment after these three probe scores arrive.
$Jobs = @(
    @{
        Slug = "rogii-c1r8-d16-b12-calibrated"
        Version = 1
        Output = "kaggle\outputs\rogii-c1r8-d16-b12-calibrated-v1"
        Message = "joint rank8 datum plus b1 b2 contrast calibration"
    },
    @{
        Slug = "rogii-c1r8-d16-a0b3"
        Version = 2
        Output = "kaggle\outputs\rogii-c1r8-d16-a0b3-v2"
        Message = "datum16 b3 Hadamard code a0 amplitude2"
    },
    @{
        Slug = "rogii-c1r8-d16-a1b3"
        Version = 2
        Output = "kaggle\outputs\rogii-c1r8-d16-a1b3-v2"
        Message = "datum16 b3 Hadamard code a1 amplitude2"
    },
    @{
        Slug = "rogii-c1r8-d16-a2b3"
        Version = 2
        Output = "kaggle\outputs\rogii-c1r8-d16-a2b3-v2"
        Message = "datum16 b3 Hadamard code a2 amplitude2"
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

Write-Log "worker ready; not-before=$($StartAt.ToString('o')) jobs=$($Jobs.Count) fifth-slot-reserved=true"
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
Write-Log "all four submissions accepted; fifth slot remains reserved"
