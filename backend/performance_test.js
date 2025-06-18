#!/usr/bin/env node
/**
 * 資料庫效能測試腳本
 * 測試索引優化前後的查詢效能
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const dbPath = path.join(__dirname, 'db.sqlite3');

class PerformanceTest {
    constructor() {
        this.db = null;
        this.testResults = [];
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.db = new sqlite3.Database(dbPath, sqlite3.OPEN_READWRITE, (err) => {
                if (err) {
                    console.error('無法連接到資料庫:', err.message);
                    reject(err);
                } else {
                    console.log('✅ 已連接到資料庫');
                    resolve();
                }
            });
        });
    }

    async runQuery(query, params = [], description = '') {
        return new Promise((resolve, reject) => {
            const startTime = process.hrtime.bigint();
            
            this.db.all(query, params, (err, rows) => {
                const endTime = process.hrtime.bigint();
                const durationMs = Number(endTime - startTime) / 1000000;
                
                if (err) {
                    console.error(`❌ 查詢失敗: ${description}`, err.message);
                    reject(err);
                } else {
                    const result = {
                        description,
                        query: query.substring(0, 100) + '...',
                        params,
                        duration: durationMs,
                        resultCount: rows.length
                    };
                    
                    this.testResults.push(result);
                    console.log(`⏱️  ${description}: ${durationMs.toFixed(2)}ms (${rows.length} 筆記錄)`);
                    resolve(rows);
                }
            });
        });
    }

    async checkIndexes() {
        console.log('\n📋 檢查現有索引...');
        
        const indexes = await this.runQuery(`
            SELECT name, sql 
            FROM sqlite_master 
            WHERE type='index' 
              AND tbl_name='semantic_clustering_sentiment'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        `, [], '索引列表查詢');
        
        console.log('現有索引:');
        indexes.forEach(idx => {
            console.log(`  - ${idx.name}`);
        });
        
        return indexes;
    }

    async runPerformanceTests() {
        console.log('\n🧪 開始效能測試...');
        
        // 測試1: 日期範圍查詢
        await this.runQuery(`
            SELECT COUNT(*) as count
            FROM semantic_clustering_sentiment 
            WHERE date(createdAt) BETWEEN '2022-01-01' AND '2022-12-31'
        `, [], '日期範圍查詢 (全年)');
        
        // 測試2: 特定日期查詢
        await this.runQuery(`
            SELECT COUNT(*) as count
            FROM semantic_clustering_sentiment 
            WHERE date(createdAt) = '2022-09-11'
        `, [], '特定日期查詢');
        
        // 測試3: 情緒過濾查詢
        await this.runQuery(`
            SELECT COUNT(*) as count
            FROM semantic_clustering_sentiment 
            WHERE sentiment = 'Positive'
        `, [], '情緒過濾查詢 (Positive)');
        
        // 測試4: 聚類查詢
        await this.runQuery(`
            SELECT COUNT(*) as count
            FROM semantic_clustering_sentiment 
            WHERE cluster_id = 1
        `, [], '聚類查詢 (cluster_id = 1)');
        
        // 測試5: 複合查詢 (日期 + 情緒)
        await this.runQuery(`
            SELECT COUNT(*) as count
            FROM semantic_clustering_sentiment 
            WHERE date(createdAt) = '2022-09-11' AND sentiment = 'Neutral'
        `, [], '複合查詢 (日期+情緒)');
        
        // 測試6: 複合查詢 (日期 + 聚類)
        await this.runQuery(`
            SELECT COUNT(*) as count
            FROM semantic_clustering_sentiment 
            WHERE date(createdAt) = '2022-09-11' AND cluster_id = 2
        `, [], '複合查詢 (日期+聚類)');
        
        // 測試7: 文本搜索
        await this.runQuery(`
            SELECT COUNT(*) as count
            FROM semantic_clustering_sentiment 
            WHERE cleaned_text LIKE '%bitcoin%'
        `, [], '文本搜索 (bitcoin)');
        
        // 測試8: 模擬實際API查詢
        await this.runQuery(`
            SELECT id, cleaned_text, createdAt, sentiment 
            FROM semantic_clustering_sentiment 
            WHERE date(createdAt) = '2022-09-11' AND sentiment = 'Neutral'
            ORDER BY createdAt DESC 
            LIMIT 50
        `, [], '模擬API查詢 (完整結果集)');
    }

    async getTableStats() {
        console.log('\n📊 資料表統計信息...');
        
        // 總記錄數
        const totalCount = await this.runQuery(`
            SELECT COUNT(*) as count FROM semantic_clustering_sentiment
        `, [], '總記錄數查詢');
        
        // 日期範圍
        const dateRange = await this.runQuery(`
            SELECT 
                MIN(date(createdAt)) as min_date,
                MAX(date(createdAt)) as max_date
            FROM semantic_clustering_sentiment
        `, [], '日期範圍查詢');
        
        // 情緒分布
        const sentimentDist = await this.runQuery(`
            SELECT sentiment, COUNT(*) as count
            FROM semantic_clustering_sentiment 
            GROUP BY sentiment
            ORDER BY count DESC
        `, [], '情緒分布查詢');
        
        // 聚類分布
        const clusterDist = await this.runQuery(`
            SELECT cluster_id, COUNT(*) as count
            FROM semantic_clustering_sentiment 
            GROUP BY cluster_id
            ORDER BY cluster_id
            LIMIT 10
        `, [], '聚類分布查詢 (前10)');
        
        console.log('\n📈 統計結果:');
        console.log(`總記錄數: ${totalCount[0].count.toLocaleString()}`);
        console.log(`日期範圍: ${dateRange[0].min_date} ~ ${dateRange[0].max_date}`);
        console.log('情緒分布:');
        sentimentDist.forEach(item => {
            console.log(`  ${item.sentiment}: ${item.count.toLocaleString()}`);
        });
        console.log('聚類分布 (前10):');
        clusterDist.forEach(item => {
            console.log(`  Cluster ${item.cluster_id}: ${item.count.toLocaleString()}`);
        });
    }

    generateReport() {
        console.log('\n📋 效能測試報告');
        console.log('=' * 50);
        
        // 按執行時間排序
        const sortedResults = this.testResults
            .filter(r => r.description !== '索引列表查詢')
            .sort((a, b) => a.duration - b.duration);
        
        console.log('\n⚡ 查詢效能排名 (快到慢):');
        sortedResults.forEach((result, index) => {
            const icon = result.duration < 10 ? '🟢' : result.duration < 50 ? '🟡' : '🔴';
            console.log(`${index + 1}. ${icon} ${result.description}: ${result.duration.toFixed(2)}ms`);
        });
        
        const avgDuration = sortedResults.reduce((sum, r) => sum + r.duration, 0) / sortedResults.length;
        const maxDuration = Math.max(...sortedResults.map(r => r.duration));
        const minDuration = Math.min(...sortedResults.map(r => r.duration));
        
        console.log('\n📊 統計摘要:');
        console.log(`平均查詢時間: ${avgDuration.toFixed(2)}ms`);
        console.log(`最快查詢時間: ${minDuration.toFixed(2)}ms`);
        console.log(`最慢查詢時間: ${maxDuration.toFixed(2)}ms`);
        
        // 效能等級評估
        if (avgDuration < 20) {
            console.log('🎉 效能等級: 優秀 (< 20ms)');
        } else if (avgDuration < 50) {
            console.log('✅ 效能等級: 良好 (20-50ms)');
        } else if (avgDuration < 100) {
            console.log('⚠️  效能等級: 普通 (50-100ms)');
        } else {
            console.log('❌ 效能等級: 需要優化 (> 100ms)');
        }
    }

    async close() {
        if (this.db) {
            this.db.close();
            console.log('🔐 資料庫連接已關閉');
        }
    }

    async run() {
        try {
            await this.connect();
            await this.checkIndexes();
            await this.getTableStats();
            await this.runPerformanceTests();
            this.generateReport();
        } catch (error) {
            console.error('❌ 測試執行失敗:', error.message);
        } finally {
            await this.close();
        }
    }
}

if (require.main === module) {
    console.log('🚀 資料庫效能測試工具');
    console.log('測試目標: semantic_clustering_sentiment 表');
    console.log('=' * 50);
    
    const test = new PerformanceTest();
    test.run().then(() => {
        console.log('\n✅ 測試完成!');
    }).catch(error => {
        console.error('❌ 測試失敗:', error);
        process.exit(1);
    });
} 