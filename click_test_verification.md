# 🖱️ SentimentChart 點擊測試驗證

## 🔧 修正的問題

### 1. **層級邏輯錯誤**
```javascript
// ❌ 修正前 - 混用 zlevel 和 z
線條: zlevel: 0
ClickableArea: z: 1  // 可能被 zlevel 覆蓋

// ✅ 修正後 - 統一使用 z 屬性
線條: z: 2
ClickableArea: z: 3  // 確保在最上層
```

### 2. **增強除錯信息**
```javascript
console.log('🖱️ SentimentChart: Chart clicked!', params);
console.log('🔍 Click details:', {
    componentType: params.componentType,
    seriesName: params.seriesName,
    name: params.name,
    value: params.value,
    dataIndex: params.dataIndex
});
```

## 🧪 測試步驟

### 1. **開啟開發者工具**
- 按 F12 打開 Console 面板
- 確保能看到日誌輸出

### 2. **點擊測試場景**

#### A. 點擊線條（直接點擊）
**預期行為**：
```
🖱️ SentimentChart: Chart clicked! {componentType: "series", seriesName: "Positive", ...}
🔍 Click details: {componentType: "series", seriesName: "Positive", name: "2022-07-15", ...}
📊 Series clicked: Positive for date: 2022-07-15
📈 Sentiment line clicked: Positive
📤 Sending filter data: {sentiment: "Positive", date: "2022-07-15"}
✅ Success feedback shown
🔄 Processing reset
```

#### B. 點擊空白區域（ClickableArea）
**預期行為**：
```
🖱️ SentimentChart: Chart clicked! {componentType: "series", seriesName: "ClickableArea", ...}
🔍 Click details: {componentType: "series", seriesName: "ClickableArea", name: "2022-07-15", ...}
📊 Series clicked: ClickableArea for date: 2022-07-15
🎯 ClickableArea hit - loading all data for date
📤 Sending filter data: {date: "2022-07-15"}
✅ Success feedback shown
🔄 Processing reset
```

#### C. 點擊 X 軸
**預期行為**：
```
🖱️ SentimentChart: Chart clicked! {componentType: "xAxis", value: "2022-07-15", ...}
🔍 Click details: {componentType: "xAxis", value: "2022-07-15", ...}
📅 X-axis clicked for date: 2022-07-15
📤 Sending filter data: {date: "2022-07-15"}
✅ Success feedback shown
🔄 Processing reset
```

### 3. **視覺反饋測試**
- 點擊後應立即看到「載入中...」提示
- 100ms 後顯示「✓ 已載入」
- ReadingPane 應更新顯示對應文章

### 4. **錯誤診斷**

#### 如果完全沒有日誌輸出：
- 檢查 onEvents 是否正確設置
- 確認 ReactECharts 版本兼容性
- 檢查 React 組件是否正確掛載

#### 如果有日誌但沒有反饋：
- 檢查 onTermSelect 是否正確傳入
- 檢查 showClickFeedback 是否正常工作
- 檢查 setIsProcessing 狀態管理

#### 如果點擊不精確：
- 調整 ClickableArea 的 barWidth (當前 80%)
- 調整 z 層級 (當前線條 z:2, ClickableArea z:3)
- 檢查 chartData 是否正確

## 🎯 成功標準

### ✅ 功能正常的標準：
1. **控制台有完整日誌輸出**
2. **視覺反饋及時顯示**
3. **ReadingPane 正確更新**
4. **點擊區域響應良好**
5. **無錯誤或警告**

### ⚠️ 需要進一步調試的情況：
1. **部分區域無法點擊**
2. **日誌輸出不完整**
3. **反饋延遲或缺失**
4. **資料更新不正確**

## 🔄 如果仍有問題

### 備選方案 1：調整 ClickableArea 配置
```javascript
{
    name: 'ClickableArea',
    type: 'bar',
    barWidth: '100%',  // 增加到 100%
    z: 5,              // 進一步提升層級
    // ...
}
```

### 備選方案 2：使用事件代理
```javascript
// 在 ReactECharts 上添加原生點擊事件
<ReactECharts 
    option={getOption()} 
    onEvents={chartEvents}
    onClick={(e) => console.log('Native click:', e)}
/>
```

### 備選方案 3：簡化點擊處理
```javascript
const chartEvents = {
    'click': onChartClick,
    'mouseover': (params) => console.log('Hover:', params.seriesName),
    'mouseout': () => console.log('Mouse out')
};
```

---

**預期結果**：修正後的 SentimentChart 應該能夠精確捕捉用戶點擊，提供即時反饋，並正確更新資料顯示。 