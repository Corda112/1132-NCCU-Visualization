# 聚類顯示問題調試指南

## 問題總結
1. **聚類分析前端沒有顯示** - 圖表不顯示
2. **顏色沒有區分類別** - 所有點顏色相同
3. **聚類統計文字白色加白底** - 看不到文字

## 已完成修復

### 1. 顏色區分問題修復
```jsx
// 修復前：依賴統計數據的顏色
color: getSentimentColor(clusterInfo.dominant_sentiment)

// 修復後：使用預設調色盤備用
const clusterColors = [
    '#ff4d4f', '#52c41a', '#1890ff', '#fa8c16', '#722ed1',
    '#eb2f96', '#13c2c2', '#a0d911', '#fa541c', '#2f54eb'
];
const clusterColor = clusterColors[clusterId % clusterColors.length];

color: clusterInfo.dominant_sentiment ? 
    getSentimentColor(clusterInfo.dominant_sentiment) : 
    clusterColor
```

### 2. 文字顏色修復
```jsx
// 修復前：可能繼承白色
textStyle: {
    color: getSentimentColor(info.dominant_sentiment),
}

// 修復後：明確深色字體
textStyle: {
    color: '#333', // 明確深色
}

// 聚類標題
color: '#262626' // 明確設定深色字體
```

### 3. 數據載入調試
```jsx
// 添加詳細日誌
console.log('ClusteringScatterPlot: Starting data fetch for range:', range);
console.log('ClusteringScatterPlot: Received cluster data:', clusterResponse.data.length, 'points');
console.log('Available clusters:', clusters);
console.log('Cluster stats:', clusterStats);
```

## 手動測試步驟

### 1. 檢查後端 API
```bash
# 啟動後端
cd backend
node server.js

# 測試 API (在另一個終端)
curl "http://localhost:5000/api/clusters?startDate=2024-01-01&endDate=2024-01-31"
curl "http://localhost:5000/api/cluster-stats?startDate=2024-01-01&endDate=2024-01-31"
```

### 2. 檢查前端控制台
```bash
# 啟動前端
cd frontend
npm start

# 在瀏覽器開發者工具查看：
# 1. Console 日誌：是否有數據載入訊息
# 2. Network 面板：API 請求是否成功
# 3. Elements 面板：圖表 SVG/Canvas 是否存在
```

### 3. 瀏覽器檢查清單
- [ ] F12 開發者工具 → Console 面板
- [ ] 查看是否有 "ClusteringScatterPlot: Received cluster data: X points" 訊息
- [ ] 查看是否有 "Available clusters: [...]" 訊息  
- [ ] Network 面板確認 `/api/clusters` 和 `/api/cluster-stats` 返回 200
- [ ] 確認圖表容器有內容（不是空白）

## 常見問題排查

### 問題 1: 圖表完全不顯示
**可能原因:**
- 數據為空：`chartData.length === 0`
- API 請求失敗
- 日期範圍無數據

**解決方案:**
1. 檢查 Console 是否有錯誤
2. 確認日期範圍包含數據
3. 檢查 API 返回是否為空陣列

### 問題 2: 點顯示但無顏色區分
**可能原因:**
- `clusterStats` 為空，無法取得 `dominant_sentiment`
- 顏色函數返回相同值

**解決方案:**
- 已修復：使用備用調色盤
- 檢查 `clusterStats` 是否載入成功

### 問題 3: 文字看不見
**可能原因:**
- CSS 顏色繼承問題
- 背景色與文字色相同

**解決方案:**
- 已修復：明確設定文字顏色為深色

## 測試驗證

### 正常顯示標準
1. ✅ 散點圖顯示多個顏色的點
2. ✅ 圖例文字清楚可見（深色字體）
3. ✅ 點擊點後能看到 tooltip
4. ✅ 統計面板文字清楚可見
5. ✅ Console 有數據載入成功訊息

### 如果仍有問題
1. 檢查資料庫是否有聚類數據
2. 確認日期範圍是否正確
3. 重新啟動後端和前端
4. 清除瀏覽器快取 