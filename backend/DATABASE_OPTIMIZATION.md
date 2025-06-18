# 資料庫效能優化指南

## 🚀 概述

本指南說明如何優化 `semantic_clustering_sentiment` 資料表的查詢效能。針對55,142條記錄的推文資料，我們實施了全面的索引策略來改善查詢速度。

## ⚡ 效能問題

### 原始問題
- ❌ `createdAt` 欄位缺乏索引 → 日期查詢緩慢
- ❌ `sentiment` 查詢無索引 → 情緒過濾效率低
- ❌ `cluster_id` 查詢無索引 → 聚類分析緩慢

### 影響範圍
```sql
-- 這些查詢之前都會進行全表掃描
WHERE date(createdAt) BETWEEN date(?) AND date(?)
WHERE sentiment = ?
WHERE cluster_id = ?
```

## 🔧 解決方案

### 1. 自動索引創建
後端服務啟動時會自動檢查並創建必要的索引：

```javascript
// server.js 會在初始化時自動執行
const indexes = [
    'idx_created_at',      // 日期查詢優化
    'idx_sentiment',       // 情緒過濾優化
    'idx_cluster_id',      // 聚類查詢優化
    'idx_date_sentiment',  // 複合查詢優化
    'idx_date_cluster',    // 聚類時間查詢優化
    'idx_text_search'      // 文本搜索優化
];
```

### 2. 手動索引管理

#### 方法一：API端點 (推薦)
```bash
# 僅在開發環境可用
curl -X POST http://localhost:3001/api/admin/create-indexes
```

#### 方法二：執行優化腳本
```bash
cd backend
node optimize_database.js
# 或
node create_indexes_api.js
```

#### 方法三：SQL腳本
```bash
sqlite3 db.sqlite3 < create_indexes.sql
```

## 📊 效能測試

### 執行測試
```bash
cd backend
node performance_test.js
```

### 測試項目
1. **日期範圍查詢** - 測試年度範圍查詢效能
2. **特定日期查詢** - 測試單日數據查詢
3. **情緒過濾查詢** - 測試情緒分類效能
4. **聚類查詢** - 測試聚類分析效能
5. **複合查詢** - 測試多條件組合查詢
6. **文本搜索** - 測試關鍵字搜索效能
7. **模擬API查詢** - 測試實際應用場景

### 預期改善
| 查詢類型 | 優化前 | 優化後 | 改善率 |
|---------|--------|--------|--------|
| 日期查詢 | >100ms | <20ms | 80%+ |
| 情緒過濾 | >50ms | <10ms | 80%+ |
| 複合查詢 | >200ms | <30ms | 85%+ |

## 🛠️ 工具說明

### 1. `optimize_database.py`
Python腳本，可獨立執行索引創建：
```bash
python optimize_database.py           # 完整優化
python optimize_database.py --info-only  # 僅查看索引信息
python optimize_database.py --vacuum     # 僅執行資料庫清理
```

### 2. `create_indexes_api.js`
Node.js腳本，通過API創建索引：
```bash
node create_indexes_api.js
```

### 3. `performance_test.js`
效能測試工具：
```bash
node performance_test.js
```

### 4. `create_indexes.sql`
純SQL腳本：
```bash
sqlite3 db.sqlite3 < create_indexes.sql
```

## 📋 索引詳細信息

### 單一欄位索引
```sql
-- 日期查詢優化 (用於時間範圍過濾)
CREATE INDEX idx_created_at ON semantic_clustering_sentiment(createdAt);

-- 情緒分析優化 (用於情緒分類過濾)
CREATE INDEX idx_sentiment ON semantic_clustering_sentiment(sentiment);

-- 聚類分析優化 (用於聚類過濾)
CREATE INDEX idx_cluster_id ON semantic_clustering_sentiment(cluster_id);

-- 文本搜索優化 (用於關鍵字搜索)
CREATE INDEX idx_text_search ON semantic_clustering_sentiment(cleaned_text);
```

### 複合索引
```sql
-- 日期+情緒複合查詢優化
CREATE INDEX idx_date_sentiment ON semantic_clustering_sentiment(createdAt, sentiment);

-- 日期+聚類複合查詢優化
CREATE INDEX idx_date_cluster ON semantic_clustering_sentiment(createdAt, cluster_id);
```

## 🔍 監控與維護

### 檢查索引狀態
```sql
SELECT name, sql 
FROM sqlite_master 
WHERE type='index' 
  AND tbl_name='semantic_clustering_sentiment'
  AND name NOT LIKE 'sqlite_%';
```

### 更新統計信息
```sql
ANALYZE semantic_clustering_sentiment;
```

### 資料庫清理
```sql
VACUUM;
```

## ⚠️ 注意事項

1. **索引空間成本**: 索引會增加約20-30%的儲存空間
2. **寫入效能**: 新增資料時會稍微影響插入速度
3. **維護頻率**: 建議每月執行一次 `ANALYZE` 更新統計信息
4. **生產環境**: 索引創建API在生產環境中被禁用

## 📈 效能基準

### 優化目標
- ✅ 平均查詢時間 < 50ms
- ✅ 日期查詢 < 20ms
- ✅ 情緒過濾 < 10ms
- ✅ 複合查詢 < 30ms

### 效能等級
- 🎉 **優秀**: < 20ms 平均查詢時間
- ✅ **良好**: 20-50ms 平均查詢時間
- ⚠️ **普通**: 50-100ms 平均查詢時間
- ❌ **需要優化**: > 100ms 平均查詢時間

## 🔧 故障排除

### 常見問題

#### 1. 索引創建失敗
```bash
# 檢查資料庫檔案權限
ls -la db.sqlite3

# 檢查磁碟空間
df -h
```

#### 2. 查詢仍然緩慢
```sql
-- 檢查查詢計劃
EXPLAIN QUERY PLAN 
SELECT * FROM semantic_clustering_sentiment 
WHERE date(createdAt) = '2022-09-11';
```

#### 3. 服務啟動時索引失敗
查看後端日誌中的索引創建警告信息，通常不會影響服務正常運行。

## 📞 技術支援

如果遇到效能問題或索引相關問題：

1. 執行效能測試: `node performance_test.js`
2. 檢查後端日誌中的索引創建信息
3. 驗證索引是否正確創建
4. 考慮執行 `VACUUM` 清理資料庫 