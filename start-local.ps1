# Start script for local development with external databases
# This script uses your local Ollama, MongoDB, and MariaDB instances

Write-Host "[START] Starting Data Analyzer with local databases..." -ForegroundColor Green

# Check if local services are running
Write-Host "[CHECK] Checking local services..." -ForegroundColor Yellow

# Check Ollama
try {
    $ollamaResponse = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
    Write-Host "[OK] Ollama is running locally" -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] Ollama is not running. Please start it first." -ForegroundColor Red
    exit 1
}

# Check MongoDB
Write-Host "[CHECK] Testing MongoDB connection..." -ForegroundColor Yellow
try {
    # Use a more reliable method to check MongoDB
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $tcpClient.ReceiveTimeout = 3000
    $tcpClient.SendTimeout = 3000
    $connection = $tcpClient.ConnectAsync("localhost", 27017)
    $connection.Wait(3000) | Out-Null
    
    if ($tcpClient.Connected) {
        Write-Host "[OK] MongoDB is running locally (Port 27017)" -ForegroundColor Green
        $tcpClient.Close()
    } else {
        Write-Host "[WARN] MongoDB connection failed - continuing anyway" -ForegroundColor Yellow
    }
}
catch {
    # Alternative check using netstat
    try {
        $netstatResult = netstat -an | Select-String ":27017.*LISTENING"
        if ($netstatResult) {
            Write-Host "[OK] MongoDB detected via netstat (Port 27017)" -ForegroundColor Green
        } else {
            Write-Host "[WARN] MongoDB not detected via netstat - continuing anyway" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "[WARN] Cannot verify MongoDB status - continuing anyway" -ForegroundColor Yellow
    }
}

# Check MariaDB
Write-Host "[CHECK] Testing MariaDB connection..." -ForegroundColor Yellow
try {
    # Use a more reliable method to check MariaDB
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $tcpClient.ReceiveTimeout = 3000
    $tcpClient.SendTimeout = 3000
    $connection = $tcpClient.ConnectAsync("localhost", 3306)
    $connection.Wait(3000) | Out-Null
    
    if ($tcpClient.Connected) {
        Write-Host "[OK] MariaDB is running locally (Port 3306)" -ForegroundColor Green
        $tcpClient.Close()
    } else {
        Write-Host "[WARN] MariaDB connection failed - continuing anyway" -ForegroundColor Yellow
    }
}
catch {
    # Alternative check using netstat
    try {
        $netstatResult = netstat -an | Select-String ":3306.*LISTENING"
        if ($netstatResult) {
            Write-Host "[OK] MariaDB detected via netstat (Port 3306)" -ForegroundColor Green
        } else {
            Write-Host "[WARN] MariaDB not detected via netstat - continuing anyway" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "[WARN] Cannot verify MariaDB status - continuing anyway" -ForegroundColor Yellow
    }
}

# Load environment variables
if (Test-Path ".env.local") {
    Write-Host "[CONFIG] Loading .env.local configuration..." -ForegroundColor Yellow
    Get-Content ".env.local" | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}
else {
    Write-Host "[WARN] .env.local not found, using defaults" -ForegroundColor Yellow
}

# Start only the dataanalyzer service
Write-Host "[DOCKER] Starting Data Analyzer container..." -ForegroundColor Yellow
docker-compose -f docker-compose.local.yml up -d dataanalyzer

# Optional: Start additional services
$jupyter = Read-Host "[PROMPT] Do you want to start Jupyter Notebook? (y/n)"
if ($jupyter -eq "y" -or $jupyter -eq "Y") {
    Write-Host "[JUPYTER] Starting Jupyter Notebook..." -ForegroundColor Yellow
    docker-compose -f docker-compose.local.yml --profile jupyter up -d jupyter
}

$metabase = Read-Host "[PROMPT] Do you want to start Metabase? (y/n)"
if ($metabase -eq "y" -or $metabase -eq "Y") {
    Write-Host "[METABASE] Starting Metabase..." -ForegroundColor Yellow
    docker-compose -f docker-compose.local.yml --profile metabase up -d metabase
}

Write-Host "[SUCCESS] Services started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "[ACCESS] Access points:" -ForegroundColor Cyan
Write-Host "   - Data Analyzer: http://localhost:7860"
Write-Host "   - Jupyter (if started): http://localhost:8888 (Token: dataanalyzer123)"
Write-Host "   - Metabase (if started): http://localhost:3000"
Write-Host ""
Write-Host "[LOGS] To check logs: docker-compose -f docker-compose.local.yml logs -f dataanalyzer" -ForegroundColor Yellow
Write-Host "[STOP] To stop: docker-compose -f docker-compose.local.yml down" -ForegroundColor Yellow
