import sqlite3
import json
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'db.sqlite3')

def add_data_to_db():
    db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create and populate semantic_clustering_sentiment table
    semantic_json_path = os.path.join(os.path.dirname(__file__), 'Final_semantic_clustering_sentiment.json')
    if os.path.exists(semantic_json_path):
        print("Processing Final_semantic_clustering_sentiment.json...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS semantic_clustering_sentiment (
            id TEXT PRIMARY KEY,
            cleaned_text TEXT,
            createdAt TEXT,
            cluster_id INTEGER,
            sentiment TEXT,
            x REAL,
            y REAL
        )
        ''')

        with open(semantic_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                cursor.execute('''
                INSERT OR REPLACE INTO semantic_clustering_sentiment (id, cleaned_text, createdAt, cluster_id, sentiment, x, y)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (item['id'], item['cleaned_text'], item['createdAt'], item['cluster_id'], item['sentiment'], item['x'], item['y']))
        print("Finished processing Final_semantic_clustering_sentiment.json.")
    else:
        print(f"File not found: {semantic_json_path}")

    # Create and populate term_ngram_frequency table
    term_json_path = os.path.join(os.path.dirname(__file__), 'Final_term_ngram_frequency.json')
    if os.path.exists(term_json_path):
        print("Processing Final_term_ngram_frequency.json...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS term_ngram_frequency (
            date TEXT,
            term TEXT,
            frequency INTEGER,
            PRIMARY KEY (date, term)
        )
        ''')
        
        with open(term_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for date, terms in data.items():
                for term, frequency in terms.items():
                    cursor.execute('''
                    INSERT OR REPLACE INTO term_ngram_frequency (date, term, frequency)
                    VALUES (?, ?, ?)
                    ''', (date, term, frequency))
        print("Finished processing Final_term_ngram_frequency.json.")
    else:
        print(f"File not found: {term_json_path}")

    conn.commit()
    conn.close()
    print("Database has been updated successfully.")

def create_performance_indexes():
    """為資料庫表創建效能索引"""
    
    # 資料庫路徑
    db_path = DB_PATH
    
    if not os.path.exists(db_path):
        print(f"❌ 資料庫文件不存在: {db_path}")
        return False
    
    try:
        # 連接資料庫
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 檢查現有索引...")
        
        # 檢查現有索引
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='semantic_clustering_sentiment'")
        existing_indexes = [row[0] for row in cursor.fetchall()]
        print(f"現有索引: {existing_indexes}")
        
        # 要創建的索引列表
        indexes_to_create = [
            {
                'name': 'idx_created_at',
                'sql': 'CREATE INDEX IF NOT EXISTS idx_created_at ON semantic_clustering_sentiment(createdAt)',
                'description': 'createdAt 欄位索引 (用於日期範圍查詢)'
            },
            {
                'name': 'idx_sentiment',
                'sql': 'CREATE INDEX IF NOT EXISTS idx_sentiment ON semantic_clustering_sentiment(sentiment)',
                'description': 'sentiment 欄位索引 (用於情緒過濾)'
            },
            {
                'name': 'idx_cluster_id',
                'sql': 'CREATE INDEX IF NOT EXISTS idx_cluster_id ON semantic_clustering_sentiment(cluster_id)',
                'description': 'cluster_id 欄位索引 (用於聚類查詢)'
            },
            {
                'name': 'idx_date_sentiment',
                'sql': 'CREATE INDEX IF NOT EXISTS idx_date_sentiment ON semantic_clustering_sentiment(createdAt, sentiment)',
                'description': '複合索引 (日期+情緒，用於組合查詢)'
            },
            {
                'name': 'idx_date_cluster',
                'sql': 'CREATE INDEX IF NOT EXISTS idx_date_cluster ON semantic_clustering_sentiment(createdAt, cluster_id)',
                'description': '複合索引 (日期+聚類，用於聚類時間查詢)'
            }
        ]
        
        # 檢查表結構
        print("\n📊 檢查表結構...")
        cursor.execute("PRAGMA table_info(semantic_clustering_sentiment)")
        columns = cursor.fetchall()
        print("表欄位:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # 檢查記錄數量
        cursor.execute("SELECT COUNT(*) FROM semantic_clustering_sentiment")
        record_count = cursor.fetchone()[0]
        print(f"\n📈 總記錄數: {record_count:,}")
        
        # 創建索引
        print("\n🔧 開始創建索引...")
        
        for index in indexes_to_create:
            try:
                start_time = time.time()
                
                print(f"創建索引: {index['name']} - {index['description']}")
                cursor.execute(index['sql'])
                
                end_time = time.time()
                print(f"  ✅ 完成 ({end_time - start_time:.2f}s)")
                
            except sqlite3.Error as e:
                print(f"  ❌ 失敗: {e}")
        
        # 提交變更
        conn.commit()
        
        # 檢查創建後的索引
        print("\n🔍 檢查創建後的索引...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='semantic_clustering_sentiment'")
        new_indexes = [row[0] for row in cursor.fetchall()]
        print(f"所有索引: {new_indexes}")
        
        # 分析表以更新統計信息
        print("\n📊 更新表統計信息...")
        cursor.execute("ANALYZE semantic_clustering_sentiment")
        conn.commit()
        
        print("\n✅ 索引優化完成!")
        
        # 測試查詢效能
        print("\n🧪 測試查詢效能...")
        test_queries = [
            {
                'name': '日期範圍查詢',
                'sql': "SELECT COUNT(*) FROM semantic_clustering_sentiment WHERE date(createdAt) BETWEEN '2022-01-01' AND '2022-12-31'"
            },
            {
                'name': '情緒過濾查詢',
                'sql': "SELECT COUNT(*) FROM semantic_clustering_sentiment WHERE sentiment = 'Positive'"
            },
            {
                'name': '複合查詢 (日期+情緒)',
                'sql': "SELECT COUNT(*) FROM semantic_clustering_sentiment WHERE date(createdAt) = '2022-09-11' AND sentiment = 'Neutral'"
            }
        ]
        
        for query in test_queries:
            start_time = time.time()
            cursor.execute(query['sql'])
            result = cursor.fetchone()[0]
            end_time = time.time()
            
            print(f"  {query['name']}: {result:,} 筆記錄 ({end_time - start_time:.3f}s)")
        
        return True
        
    except sqlite3.Error as e:
        print(f"❌ 資料庫錯誤: {e}")
        return False
    
    finally:
        if conn:
            conn.close()

def show_index_info():
    """顯示索引信息"""
    db_path = DB_PATH
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\n📋 索引詳細信息:")
        cursor.execute("""
            SELECT name, sql 
            FROM sqlite_master 
            WHERE type='index' 
            AND tbl_name='semantic_clustering_sentiment'
            AND name NOT LIKE 'sqlite_%'
        """)
        
        indexes = cursor.fetchall()
        for name, sql in indexes:
            print(f"\n索引名稱: {name}")
            print(f"SQL: {sql}")
        
    except sqlite3.Error as e:
        print(f"❌ 錯誤: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # 首先檢查是否只需要創建索引
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--index-only":
        print("🚀 開始資料庫索引優化...")
        print("=" * 50)
        
        success = create_performance_indexes()
        
        if success:
            show_index_info()
            print("\n🎉 資料庫索引優化完成!")
            print("\n📌 建議:")
            print("  1. 重新啟動後端服務以獲得最佳效能")
            print("  2. 監控查詢時間是否有所改善")
            print("  3. 定期執行 ANALYZE 指令更新統計信息")
        else:
            print("\n❌ 優化失敗，請檢查錯誤信息")
    else:
        # 原本的資料導入流程
        add_data_to_db()
