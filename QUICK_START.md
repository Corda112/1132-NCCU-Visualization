# ⚡ 快速部署指南

您的部署遇到了計費帳戶問題。這是 Google Cloud 的標準要求，即使使用免費額度也需要設定。

## 🎯 立即解決方案

### 第 1 步：啟用計費（5 分鐘）

1. **開啟計費頁面**：
   ```
   https://console.cloud.google.com/billing
   ```

2. **創建計費帳戶**：
   - 選擇 "個人" 帳戶
   - 輸入信用卡資訊（只是驗證，不會收費）
   - 連結到專案：`vis-1132-nccu-crypto`

3. **驗證設定**：
   ```powershell
   gcloud billing projects describe vis-1132-nccu-crypto
   ```

### 第 2 步：重新部署

```powershell
.\deploy.ps1 -ProjectId vis-1132-nccu-crypto
```

## 💰 費用保障

- ✅ **完全免費**：您的用量遠低於免費額度
- ✅ **安全保障**：可設定 $1 預算警報
- ✅ **永久免費**：每月 200 萬次請求額度

## 📋 部署將會：

1. ✅ 打包前端 React 應用
2. ✅ 整合到 Node.js 後端
3. ✅ 部署到 Google Cloud Run
4. ✅ 提供公開 URL
5. ✅ 自動擴展和管理

## 🚀 部署完成後

您將獲得：
- 🌐 **公開網址**：例如 `https://bitcoin-sentiment-app-xxx.run.app`
- 📱 **響應式界面**：手機、平板、電腦都能使用
- 🔄 **自動備份**：Google 提供 99.9% 可用性
- 📊 **免費監控**：實時日誌和性能監控

## ❓ 需要幫助？

- 📖 **詳細指南**：`BILLING_SETUP.md`
- 🚀 **部署說明**：`DEPLOYMENT.md`
- 🛠️ **技術問題**：檢查 `README.md`

## ⏱️ 預計時間

- 設定計費：5 分鐘
- 執行部署：10-15 分鐘
- **總計**：約 20 分鐘完成上雲！

---

💡 **提示**：這是一次性設定，之後部署只需要執行一個指令！ 