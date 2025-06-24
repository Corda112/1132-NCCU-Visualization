-- 資料庫效能優化 - 索引創建腳本
-- 為 semantic_clustering_sentiment 表添加必要的索引

-- 1. createdAt 欄位索引 (用於日期範圍查詢)
CREATE INDEX IF NOT EXISTS idx_created_at ON semantic_clustering_sentiment(createdAt);

-- 2. sentiment 欄位索引 (用於情緒過濾)
CREATE INDEX IF NOT EXISTS idx_sentiment ON semantic_clustering_sentiment(sentiment);

-- 3. cluster_id 欄位索引 (用於聚類查詢)
CREATE INDEX IF NOT EXISTS idx_cluster_id ON semantic_clustering_sentiment(cluster_id);

-- 4. 複合索引 (日期+情緒，用於組合查詢)
CREATE INDEX IF NOT EXISTS idx_date_sentiment ON semantic_clustering_sentiment(createdAt, sentiment);

-- 5. 複合索引 (日期+聚類，用於聚類時間查詢)
CREATE INDEX IF NOT EXISTS idx_date_cluster ON semantic_clustering_sentiment(createdAt, cluster_id);

-- 6. cleaned_text 索引 (用於文本搜索優化)
CREATE INDEX IF NOT EXISTS idx_text_search ON semantic_clustering_sentiment(cleaned_text);

-- 更新表統計信息以提升查詢優化器效率
ANALYZE semantic_clustering_sentiment;

-- 檢查創建的索引
SELECT 
    name as index_name,
    sql as index_definition
FROM sqlite_master 
WHERE type='index' 
    AND tbl_name='semantic_clustering_sentiment'
    AND name NOT LIKE 'sqlite_%'
ORDER BY name; 