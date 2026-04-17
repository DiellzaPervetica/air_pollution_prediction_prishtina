$baseFolder = "C:\Users\Festina\Desktop\openaq_prishtina"

if (!(Test-Path $baseFolder)) {
    New-Item -ItemType Directory -Path $baseFolder | Out-Null
}

$locations = @(2536, 7674, 7931, 7933, 9337)

foreach ($loc in $locations) {
    $dest = Join-Path $baseFolder $loc

    if (!(Test-Path $dest)) {
        New-Item -ItemType Directory -Path $dest | Out-Null
    }

    Write-Host "Downloading location ID $loc ..."
    aws s3 cp "s3://openaq-data-archive/records/csv.gz/locationid=$loc/" $dest --recursive --no-sign-request
    Write-Host "Finished location ID $loc"
}

Write-Host "Te gjitha te dhenat per Prishtinen u shkarkuan!"