#!/usr/bin/env python3
"""
資料庫效能優化腳本
專門用於為 semantic_clustering_sentiment 表添加必要的索引
"""

import sqlite3
import os
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'db.sqlite3')

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
            },
            {
                'name': 'idx_text_search',
                'sql': 'CREATE INDEX IF NOT EXISTS idx_text_search ON semantic_clustering_sentiment(cleaned_text)',
                'description': 'cleaned_text 索引 (用於文本搜索優化)'
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
            },
            {
                'name': '聚類查詢',
                'sql': "SELECT COUNT(*) FROM semantic_clustering_sentiment WHERE cluster_id = 1"
            },
            {
                'name': '文本搜索',
                'sql': "SELECT COUNT(*) FROM semantic_clustering_sentiment WHERE cleaned_text LIKE '%bitcoin%'"
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
        
        # 顯示索引使用統計
        print("\n📊 索引使用統計:")
        cursor.execute("PRAGMA index_list(semantic_clustering_sentiment)")
        index_list = cursor.fetchall()
        
        for index_info in index_list:
            index_name = index_info[1]
            if not index_name.startswith('sqlite_'):
                try:
                    cursor.execute(f"PRAGMA index_info({index_name})")
                    index_details = cursor.fetchall()
                    print(f"  {index_name}: {len(index_details)} 欄位")
                except:
                    pass
        
    except sqlite3.Error as e:
        print(f"❌ 錯誤: {e}")
    finally:
        if conn:
            conn.close()

def vacuum_database():
    """清理和壓縮資料庫"""
    db_path = DB_PATH
    
    try:
        print("\n🧹 開始資料庫清理...")
        conn = sqlite3.connect(db_path)
        
        # 檢查資料庫大小
        file_size = os.path.getsize(db_path)
        print(f"清理前大小: {file_size / 1024 / 1024:.2f} MB")
        
        # 執行 VACUUM
        conn.execute("VACUUM")
        conn.close()
        
        # 檢查清理後大小
        new_file_size = os.path.getsize(db_path)
        print(f"清理後大小: {new_file_size / 1024 / 1024:.2f} MB")
        print(f"節省空間: {(file_size - new_file_size) / 1024 / 1024:.2f} MB")
        
    except sqlite3.Error as e:
        print(f"❌ 清理失敗: {e}")

if __name__ == "__main__":
    print("🚀 資料庫效能優化工具")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--info-only":
        # 只顯示索引信息
        show_index_info()
    elif len(sys.argv) > 1 and sys.argv[1] == "--vacuum":
        # 只執行清理
        vacuum_database()
    else:
        # 完整優化流程
        success = create_performance_indexes()
        
        if success:
            show_index_info()
            
            # 詢問是否執行清理
            response = input("\n是否執行資料庫清理 (VACUUM)? [y/N]: ")
            if response.lower() in ['y', 'yes']:
                vacuum_database()
            
            print("\n🎉 資料庫效能優化完成!")
            print("\n📌 建議:")
            print("  1. 重新啟動後端服務以獲得最佳效能")
            print("  2. 監控查詢時間是否有所改善")
            print("  3. 定期執行 ANALYZE 指令更新統計信息")
            print("  4. 使用 --info-only 參數查看索引狀態")
            print("  5. 使用 --vacuum 參數清理資料庫")
        else:
            print("\n❌ 優化失敗，請檢查錯誤信息") 