#!/bin/bash

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 開始部署 Bitcoin 情緒分析系統到 Google Cloud Run...${NC}"

# 檢查是否已安裝 gcloud
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Google Cloud CLI 未安裝，請先安裝: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# 設定預設值
PROJECT_ID=${1:-""}
SERVICE_NAME=${2:-"bitcoin-sentiment-app"}
REGION=${3:-"us-central1"}

# 如果沒有提供 PROJECT_ID，嘗試從 gcloud 獲取
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}❌ 請提供 Google Cloud Project ID${NC}"
        echo -e "${YELLOW}使用方式: ./deploy.sh PROJECT_ID [SERVICE_NAME] [REGION]${NC}"
        echo -e "${YELLOW}或先設定預設專案: gcloud config set project YOUR_PROJECT_ID${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}📋 部署參數:${NC}"
echo -e "  Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "  Service Name: ${GREEN}$SERVICE_NAME${NC}"
echo -e "  Region: ${GREEN}$REGION${NC}"

read -p "確定要繼續部署嗎? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🚫 部署已取消${NC}"
    exit 0
fi

# 設定專案
echo -e "${BLUE}🔧 設定 Google Cloud 專案...${NC}"
gcloud config set project $PROJECT_ID

# 啟用必要的 API
echo -e "${BLUE}🔌 啟用必要的 Google Cloud APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# 部署到 Cloud Run
echo -e "${BLUE}🚀 部署到 Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="NODE_ENV=production" \
  --memory=512Mi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=10

if [ $? -eq 0 ]; then
    # 獲取服務 URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    
    echo -e "${GREEN}🎉 部署成功！${NC}"
    echo -e "${GREEN}🌐 應用程式 URL: ${SERVICE_URL}${NC}"
    echo -e "${GREEN}🔍 健康檢查: ${SERVICE_URL}/health${NC}"
    echo -e "${BLUE}📊 管理服務: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME${NC}"
    
    # 測試健康檢查
    echo -e "${BLUE}🔍 測試健康檢查...${NC}"
    sleep 10
    if curl -s "${SERVICE_URL}/health" > /dev/null; then
        echo -e "${GREEN}✅ 服務正常運行${NC}"
    else
        echo -e "${YELLOW}⚠️ 服務可能還在啟動中，請稍後再試${NC}"
    fi
else
    echo -e "${RED}❌ 部署失敗${NC}"
    exit 1
fi 