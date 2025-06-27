# 💳 Google Cloud 計費設定指南

## ❗ 重要提醒

即使使用 Google Cloud 的**免費額度**，也必須設定計費帳戶才能啟用 Cloud Run 等服務。**在免費額度內不會產生任何費用**。

## 🚀 快速設定步驟

### 1. 前往計費頁面

在瀏覽器中開啟：
```
https://console.cloud.google.com/billing
```

### 2. 創建計費帳戶

1. 點選 **"創建帳戶"** 或 **"管理計費帳戶"**
2. 選擇 **"個人"** 帳戶類型
3. 填寫基本資訊：
   - 國家/地區
   - 姓名
   - 地址
4. 輸入信用卡或金融卡資訊
   - ⭐ **重要**：只是驗證身份，免費額度內不會收費
   - 建議使用有少量餘額的卡片

### 3. 將計費帳戶連結到專案

1. 選擇您的專案：`vis-1132-nccu-crypto`
2. 點選 **"連結計費帳戶"**
3. 選擇剛創建的計費帳戶
4. 確認連結

### 4. 驗證設定

```bash
# 檢查專案計費狀態
gcloud billing projects describe vis-1132-nccu-crypto
```

應該會看到：
```yaml
billingAccountName: billingAccounts/XXXXXX-XXXXXX-XXXXXX
billingEnabled: true
name: projects/vis-1132-nccu-crypto
projectId: vis-1132-nccu-crypto
```

## 💰 免費額度保障

### 永久免費額度
- **Cloud Run**: 每月 200 萬次請求
- **Cloud Build**: 每月 120 建置分鐘
- **儲存空間**: 0.5 GB

### 設定預算警報
1. 前往：https://console.cloud.google.com/billing/budgets
2. 點選 **"創建預算"**
3. 設定金額：**$1 USD**（遠低於免費額度）
4. 設定警報：**50%, 90%, 100%**
5. 這樣即使意外超過免費額度，也會立即收到通知

## 🛡️ 安全建議

### 1. 設定花費限制
```bash
# 停用超過預算時的服務（可選）
gcloud billing budgets create \
    --billing-account=YOUR-BILLING-ACCOUNT \
    --display-name="Free Tier Protection" \
    --budget-amount=1USD \
    --threshold-rules=percent=100
```

### 2. 定期檢查
- 每週檢查計費報告
- 監控服務使用量
- 設定手機簡訊警報

## 🔄 完成設定後

設定完計費帳戶後，重新執行部署：

```powershell
.\deploy.ps1 -ProjectId vis-1132-nccu-crypto
```

## ❓ 常見問題

**Q: 為什麼免費服務需要信用卡？**
A: Google 需要驗證您的身份，防止濫用。在免費額度內絕對不會收費。

**Q: 如何確保不會被收費？**
A: 設定 $1 預算警報，並定期檢查使用量。您的專案用量很難超過免費額度。

**Q: 可以用 Debit Card 嗎？**
A: 可以，但建議餘額至少有 $1 用於驗證。

**Q: 如何移除計費帳戶？**
A: 專案部署完成後，可以在 Google Cloud Console 中移除計費帳戶，但服務會停止運行。

## 🎯 下一步

計費設定完成後，請返回執行：

```powershell
.\deploy.ps1 -ProjectId vis-1132-nccu-crypto
```

您的 Bitcoin 情緒分析系統將成功部署到雲端！🚀 