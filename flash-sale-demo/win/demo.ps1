# Flash-sale demo for Windows 11 + Podman Desktop.
#
#   start  download the images if needed, start three containers, load the data
#   seed   reload the data into the running containers (resets the demo)
#   stop   remove the three containers

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('start', 'seed', 'stop')]
    [string]$Action
)

$Root  = Split-Path -Parent $PSScriptRoot
$Pg    = 'flashsale-postgres'
$Mongo = 'flashsale-mongo'
$Neo   = 'flashsale-neo4j'
$Images  = @(
    'docker.io/library/postgres:17',
    'docker.io/library/mongo:8.0',
    'docker.io/library/neo4j:5-community'
)

# Recent WSL versions and Podman 6 disagree about cgroups, and containers fail to
# start with: crun: ... cgroup controller `pids` is not available.
# Starting the containers without cgroups avoids it. Cgroups only limit CPU and
# memory use, which this demo does not need.
$RunOptions = @('--cgroups=disabled')

function Fail([string]$Message) {
    Write-Host ''
    Write-Host "PROBLEM: $Message" -ForegroundColor Red
    exit 1
}

function Assert-Podman {
    if (-not (Get-Command podman -ErrorAction SilentlyContinue)) {
        Fail 'The "podman" command was not found. Install Podman Desktop, let it set up Podman, then open a new terminal.'
    }
    podman info *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail 'Podman is installed but its machine is not running. Open Podman Desktop and start the Podman machine, or run: podman machine start'
    }
}

function Show-NetworkHelp {
    Write-Host ''
    Write-Host 'PROBLEM: Podman could not download an image.' -ForegroundColor Red
    Write-Host 'If the message above says "Temporary failure in name resolution" or "dial tcp",'
    Write-Host 'the Podman machine has no working internet connection. This is a network'
    Write-Host 'setting on this PC, not a fault in the demo. Try these in order:'
    Write-Host ''
    Write-Host '  1. If a VPN is connected, disconnect it and run start.cmd again.'
    Write-Host '  2. Restart the Podman machine. In a terminal:'
    Write-Host '         podman machine stop'
    Write-Host '         wsl --shutdown'
    Write-Host '         podman machine start'
    Write-Host '     then run start.cmd again.'
    Write-Host '  3. Set a DNS server inside the Podman machine. Run:  podman machine ssh'
    Write-Host '     and at the Linux prompt:'
    Write-Host '         sudo rm /etc/resolv.conf'
    Write-Host '         echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf'
    Write-Host '         exit'
    Write-Host '     This setting is lost when the Podman machine restarts, so repeat it if needed.'
    Write-Host '  4. Test on its own:   podman pull docker.io/library/postgres:17'
    exit 1
}

function Get-Images {
    # Download each image separately so a network problem is reported clearly.
    foreach ($image in $Images) {
        podman image exists $image
        if ($LASTEXITCODE -eq 0) { continue }
        Write-Host "Downloading $image ..."
        podman pull $image
        if ($LASTEXITCODE -ne 0) { Show-NetworkHelp }
    }
}

function Remove-Containers {
    podman rm --force --volumes --ignore $Pg $Mongo $Neo *> $null
}

function Start-Containers {
    Get-Images
    Write-Host 'Starting the three databases...'
    Remove-Containers

    podman run -d @RunOptions --name $Pg -e POSTGRES_PASSWORD=demo -e POSTGRES_DB=flashsale $Images[0]
    if ($LASTEXITCODE -ne 0) { Fail 'Postgres did not start. The message above says why.' }

    podman run -d @RunOptions --name $Mongo $Images[1]
    if ($LASTEXITCODE -ne 0) { Fail 'MongoDB did not start. The message above says why.' }

    # Ports 7474 and 7687 are for Neo4j Browser (the graph pictures).
    podman run -d @RunOptions --name $Neo -e NEO4J_AUTH=none -p 7474:7474 -p 7687:7687 $Images[2]
    if ($LASTEXITCODE -ne 0) { Fail 'Neo4j did not start. If the message above mentions port 7474 or 7687, another program is using that port.' }
}

function Wait-For([string]$Name, [string]$Container, [scriptblock]$Probe) {
    Write-Host "  waiting for $Name " -NoNewline
    for ($i = 0; $i -lt 90; $i++) {
        & $Probe *> $null
        if ($LASTEXITCODE -eq 0) { Write-Host ' ready'; return }
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds 2
    }
    Fail "$Name did not become ready. See why with: podman logs $Container"
}

function Copy-In([string]$Container, [string]$Folder, [string]$Filter, [string]$Dest) {
    podman exec -u root $Container mkdir -p $Dest
    if ($LASTEXITCODE -ne 0) { Fail "Could not prepare $Dest in $Container. Is it running? Try start.cmd." }
    $files = Get-ChildItem -Path (Join-Path $Root $Folder) -Filter $Filter -File
    if (-not $files) { Fail "No $Filter files found in the $Folder folder." }
    # Copy using relative paths: a Windows path such as C:\demo has a colon in it,
    # and "podman cp" also uses a colon to separate the container from the path.
    Push-Location (Join-Path $Root $Folder)
    try {
        foreach ($file in $files) {
            podman cp "./$($file.Name)" "${Container}:$Dest/"
            if ($LASTEXITCODE -ne 0) { Fail "Could not copy $($file.Name) into $Container." }
        }
    } finally {
        Pop-Location
    }
    podman exec -u root $Container chmod -R a+rX $Dest
}

function Seed-Data {
    Write-Host ''
    Write-Host 'Checking the databases are ready...'
    Wait-For 'Postgres' $Pg    { podman exec $Pg pg_isready -h 127.0.0.1 -U postgres -d flashsale }
    Wait-For 'MongoDB'  $Mongo { podman exec $Mongo mongosh --quiet --eval 'db.runCommand({ ping: 1 })' }
    Wait-For 'Neo4j'    $Neo   { podman exec $Neo cypher-shell 'RETURN 1;' }

    Write-Host ''
    Write-Host 'Copying the shared dataset into each container...'
    Copy-In $Pg    'data'     '*.csv'       '/dataset'
    Copy-In $Pg    'postgres' '00_load.sql' '/scripts'
    Copy-In $Mongo 'data'     '*.jsonl'     '/dataset'
    Copy-In $Mongo 'mongo'    '00_load.js'  '/scripts'
    Copy-In $Neo   'data'     '*.csv'       '/var/lib/neo4j/import'
    Copy-In $Neo   'neo4j'    '00_load.cypher' '/scripts'

    Write-Host ''
    Write-Host 'Postgres (relational schema):' -ForegroundColor Cyan
    podman exec $Pg psql -q -U postgres -d flashsale -f /scripts/00_load.sql
    if ($LASTEXITCODE -ne 0) { Fail 'The Postgres load failed. The message above says why.' }

    Write-Host ''
    Write-Host 'MongoDB (documents):' -ForegroundColor Cyan
    podman exec $Mongo mongosh --quiet flashsale --file /scripts/00_load.js
    if ($LASTEXITCODE -ne 0) { Fail 'The MongoDB load failed. The message above says why.' }

    Write-Host ''
    Write-Host 'Neo4j (graph):' -ForegroundColor Cyan
    podman exec $Neo cypher-shell -f /scripts/00_load.cypher
    if ($LASTEXITCODE -ne 0) { Fail 'The Neo4j load failed. The message above says why.' }

    Write-Host ''
    Write-Host 'Done. All three should report 2000 users, 150 products, 20000 orders,' -ForegroundColor Green
    Write-Host '56487 order lines and 16255674.43 total revenue.' -ForegroundColor Green
    Write-Host 'Neo4j Browser: http://localhost:7474  (choose "No authentication")'
}

Assert-Podman
switch ($Action) {
    'start' { Start-Containers; Seed-Data }
    'seed'  { Seed-Data }
    'stop'  { Remove-Containers; Write-Host 'The three demo containers have been removed.' }
}
