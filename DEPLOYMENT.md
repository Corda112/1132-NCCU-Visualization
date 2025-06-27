# 🚀 雲端部署指南

## 一鍵部署到 Google Cloud Run

本指南將幫助您將 Bitcoin 情緒分析系統部署到 Google Cloud Run，享受 **完全免費** 的額度。

### 🎯 部署方案優勢

- ✅ **完全免費**: 使用 Google Cloud 永久免費額度
- ✅ **極度簡單**: 一個指令完成部署
- ✅ **自動擴展**: 根據流量自動調整資源
- ✅ **統一服務**: 前後端在同一域名，無 CORS 問題

### 📋 事前準備

1. **安裝 Google Cloud CLI**
   ```bash
   # Windows (使用 PowerShell)
   (New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
   & $env:Temp\GoogleCloudSDKInstaller.exe
   
   # macOS
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Linux
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   ```

2. **登入 Google Cloud**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **創建 Google Cloud 專案**（如果還沒有）
   ```bash
   # 創建新專案
   gcloud projects create YOUR-PROJECT-ID
   
       # 設定為預設專案
    gcloud config set project YOUR-PROJECT-ID
   
   # 啟用計費（免費額度也需要）
   # 請到 Google Cloud Console 手動啟用計費
   ```

### 🚀 部署步驟

#### 方法一：一鍵部署（推薦）

```bash
# 直接部署（如果已設定預設專案）
./deploy.sh

# 或指定專案 ID
./deploy.sh YOUR-PROJECT-ID

# 或完整指定所有參數
./deploy.sh YOUR-PROJECT-ID bitcoin-sentiment-app us-central1
```

#### 方法二：手動部署

```bash
# 1. 設定專案
gcloud config set project YOUR-PROJECT-ID

# 2. 啟用 APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# 3. 部署
gcloud run deploy bitcoin-sentiment-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="NODE_ENV=production" \
  --memory=512Mi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=10
```

### 🌐 部署完成

部署完成後，您會看到：

```
Service [bitcoin-sentiment-app] revision [bitcoin-sentiment-app-00001-xxx] has been deployed and is serving 100 percent of traffic.
Service URL: https://bitcoin-sentiment-app-xxxxxxxx-uc.a.run.app
```

您的應用程式現在可以通過這個 URL 訪問了！

### 🔍 測試部署

```bash
# 測試健康檢查
curl https://YOUR-SERVICE-URL/health

# 測試 API
curl https://YOUR-SERVICE-URL/api/kline
```

### 💰 免費額度說明

Google Cloud Run 提供非常慷慨的永久免費額度：

- **請求數**: 每月 200 萬次請求
- **記憶體**: 每月 360,000 GB-秒
- **CPU**: 每月 180,000 vCPU-秒
- **網路**: 每月 1GB 出站流量

對於個人專案或小型應用，這些額度幾乎不可能用完，所以可以 **完全免費** 運行。

### 🛠️ 管理服務

#### 查看服務狀態
```bash
gcloud run services list
gcloud run services describe bitcoin-sentiment-app --region=us-central1
```

#### 查看日誌
```bash
gcloud logs tail --service=bitcoin-sentiment-app
```

#### 更新服務
```bash
# 重新部署
./deploy.sh YOUR-PROJECT-ID

# 或手動更新
gcloud run deploy bitcoin-sentiment-app --source .
```

#### 刪除服務
```bash
gcloud run services delete bitcoin-sentiment-app --region=us-central1
```

### 📊 監控與管理

1. **Google Cloud Console**: https://console.cloud.google.com/run
2. **服務詳情**: https://console.cloud.google.com/run/detail/us-central1/bitcoin-sentiment-app
3. **日誌查看**: https://console.cloud.google.com/logs

### 🔧 自訂設定

如需修改部署設定，可以編輯 `cloud-run.yaml` 檔案或在部署指令中調整參數：

```bash
gcloud run deploy bitcoin-sentiment-app \
  --source . \
  --memory=1Gi \          # 增加記憶體
  --cpu=2 \               # 增加 CPU
  --max-instances=20 \    # 增加最大實例數
  --timeout=600           # 增加請求超時時間
```

### ❓ 常見問題

**Q: 為什麼需要啟用計費？**
A: 即使使用免費額度，Google Cloud 也需要有效的付款方式。在免費額度內不會收費。

**Q: 如何確保不超過免費額度？**
A: 可以在 Google Cloud Console 設定預算警報，當接近免費額度時會收到通知。

**Q: 資料庫會持久保存嗎？**
A: SQLite 檔案會保存在容器中，但重新部署時會重置。生產環境建議使用 Cloud SQL。

**Q: 如何添加自訂域名？**
A: 在 Cloud Run 服務中可以映射自訂域名，詳見 Google Cloud 文檔。

### 🎉 完成！

恭喜！您的 Bitcoin 情緒分析系統現在已經在雲端運行了。享受您的免費雲端應用程式！ 