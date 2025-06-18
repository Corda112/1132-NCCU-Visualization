# 1132-NCCU-Visualization

Bitcoin 情緒分析視覺化系統 - 結合社群媒體情緒分析、K線圖表和時間序列分解的綜合分析平台。

## 🚀 快速啟動

### 選項1: 一鍵啟動 (推薦)
```bash
# 安裝所有依賴
npm run install:all

# 同時啟動前後端 (開發模式)
npm run dev

# 或者同時啟動前後端 (生產模式)
npm start
```

### 選項2: 分別啟動
```bash
# 僅啟動後端 (生產模式)
npm run backend

# 僅啟動後端 (開發模式，自動重啟)
npm run backend:dev

# 僅啟動前端
npm run frontend
```

### 選項3: 傳統方式
```bash
# 後端
cd backend
npm install
node server.js

# 前端 (新終端機)
cd frontend
npm install
npm start
```

## 🏗️ 專案架構

```
1132-NCCU-Visualization/
├── backend/                 # Node.js + Express API 服務
│   ├── middleware/         # 安全性和驗證中間件
│   │   ├── security.js    # 速率限制、SQL注入防護
│   │   └── validation.js  # 輸入驗證
│   ├── server.js          # 主要API服務器
│   ├── db.sqlite3         # SQLite資料庫
│   └── .env               # 環境變數配置
├── frontend/               # React前端應用
│   ├── src/
│   │   ├── components/    # 視覺化組件
│   │   │   ├── KLineChart.jsx        # D3 K線圖
│   │   │   ├── UASTLChart.jsx        # D3 時序分解圖
│   │   │   ├── SentimentChart.jsx    # ECharts 情緒圖表
│   │   │   ├── FrequencyChart.jsx    # ECharts 詞頻圖表
│   │   │   └── ClusteringScatterPlot.jsx # ECharts 聚類圖
│   │   └── config/
│   │       └── api.js     # API配置和錯誤處理
└── SECURITY.md            # 安全性文件

```

## 🛡️ 安全性特色

- ✅ **SQL注入防護**: 參數化查詢 + 危險模式檢測
- ✅ **速率限制**: API全域限制 + 搜尋API特殊限制  
- ✅ **輸入驗證**: 嚴格的參數格式和長度驗證
- ✅ **安全標頭**: Helmet中間件 + CORS配置
- ✅ **錯誤處理**: 統一格式 + 敏感資訊隱藏

## 📊 功能特色

### 前端視覺化
- **D3.js**: K線圖表、UASTL時序分解、互動式刷選
- **ECharts**: 情緒分析、詞頻統計、聚類散點圖
- **React**: 響應式界面、組件化架構

### 後端API
- **K線資料**: `/api/kline` - 價格歷史資料
- **情緒分析**: `/api/semantic` - 社群情緒統計
- **詞頻分析**: `/api/term-ngram` - 關鍵詞和N-gram頻率
- **聚類資料**: `/api/clusters` - 語義聚類座標
- **文章檢索**: `/api/articles` - 相關文章分頁查詢
- **健康檢查**: `/health` - 系統狀態監控

## 🔧 可用指令

| 指令 | 說明 |
|------|------|
| `npm run backend` | 啟動後端服務器 |
| `npm run backend:dev` | 啟動後端 (開發模式) |
| `npm run frontend` | 啟動前端應用 |
| `npm run dev` | 同時啟動前後端 (開發模式) |
| `npm run start` | 同時啟動前後端 (生產模式) |
| `npm run install:all` | 安裝前後端所有依賴 |
| `npm run build` | 建置前端生產版本 |

## 📱 訪問地址

- **前端**: http://localhost:3000
- **後端API**: http://localhost:3001
- **健康檢查**: http://localhost:3001/health

## ⚙️ 環境需求

- Node.js >= 16.0.0
- npm >= 8.0.0
- SQLite3

## 📖 更多資訊

詳細的安全性配置和部署指南請參考 [SECURITY.md](SECURITY.md)