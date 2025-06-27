# PowerShell Deploy Script for Bitcoin Sentiment Analysis System
param(
    [string]$ProjectId = "",
    [string]$ServiceName = "bitcoin-sentiment-app",
    [string]$Region = "us-central1"
)

# Color output function
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Blue "🚀 Starting deployment of Bitcoin Sentiment Analysis System to Google Cloud Run..."

# Check if gcloud is installed
try {
    gcloud version | Out-Null
} catch {
    Write-ColorOutput Red "❌ Google Cloud CLI not installed. Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
}

# Get PROJECT_ID if not provided
if ([string]::IsNullOrEmpty($ProjectId)) {
    try {
        $ProjectId = gcloud config get-value project 2>$null
        if ([string]::IsNullOrEmpty($ProjectId)) {
            throw "No project set"
        }
    } catch {
        Write-ColorOutput Red "❌ Please provide Google Cloud Project ID"
        Write-ColorOutput Yellow "Usage: .\deploy.ps1 -ProjectId YOUR_PROJECT_ID [-ServiceName SERVICE_NAME] [-Region REGION]"
        Write-ColorOutput Yellow "Or set default project first: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    }
}

Write-ColorOutput Blue "📋 Deployment Parameters:"
Write-ColorOutput Green "  Project ID: $ProjectId"
Write-ColorOutput Green "  Service Name: $ServiceName"
Write-ColorOutput Green "  Region: $Region"

$response = Read-Host "Continue with deployment? (y/N)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-ColorOutput Yellow "🚫 Deployment cancelled"
    exit 0
}

# Set project
Write-ColorOutput Blue "🔧 Setting Google Cloud project..."
gcloud config set project $ProjectId

# Check billing status first
Write-ColorOutput Blue "💳 Checking billing status..."
$billingStatus = gcloud billing projects describe $ProjectId --format="value(billingEnabled)" 2>$null
if ($billingStatus -ne "True") {
    Write-ColorOutput Red "❌ Billing is not enabled for this project."
    Write-ColorOutput Yellow "Please follow the guide in BILLING_SETUP.md to enable billing."
    Write-ColorOutput Yellow "URL: https://console.cloud.google.com/billing"
    exit 1
}
Write-ColorOutput Green "✅ Billing is enabled"

# Enable required APIs
Write-ColorOutput Blue "🔌 Enabling required Google Cloud APIs..."
$apiResult1 = gcloud services enable cloudbuild.googleapis.com 2>&1
$apiResult2 = gcloud services enable run.googleapis.com 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "❌ Failed to enable APIs. Please check billing setup."
    Write-ColorOutput Yellow "See BILLING_SETUP.md for detailed instructions."
    exit 1
}

# Deploy to Cloud Run
Write-ColorOutput Blue "🚀 Deploying to Cloud Run..."
$result = gcloud run deploy $ServiceName `
    --source . `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --set-env-vars="NODE_ENV=production" `
    --memory=512Mi `
    --cpu=1 `
    --timeout=300 `
    --max-instances=10
if ($LASTEXITCODE -eq 0) {
    # Get service URL
    $ServiceUrl = gcloud run services describe $ServiceName --region=$Region --format="value(status.url)"
    
    Write-ColorOutput Green "🎉 Deployment successful!"
    Write-ColorOutput Green "🌐 Application URL: $ServiceUrl"
    Write-ColorOutput Green "🔍 Health Check: $ServiceUrl/health"
    Write-ColorOutput Blue "📊 Manage Service: https://console.cloud.google.com/run/detail/$Region/$ServiceName"
    
    # Test health check
    Write-ColorOutput Blue "🔍 Testing health check..."
    Start-Sleep -Seconds 10
    try {
        $healthCheck = Invoke-RestMethod -Uri "$ServiceUrl/health" -Method Get -TimeoutSec 10
        Write-ColorOutput Green "✅ Service is running normally"
    } catch {
        Write-ColorOutput Yellow "⚠️ Service might still be starting up, please try again later"
    }
} else {
    Write-ColorOutput Red "❌ Deployment failed"
    exit 1
} 