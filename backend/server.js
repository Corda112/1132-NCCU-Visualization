require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// 導入安全性和驗證中間件
const { apiLimiter, searchLimiter, sanitizeInput, requestSizeLimit } = require('./middleware/security');
const { validateArticleSearch, validateDateRange, validateClusterQuery } = require('./middleware/validation');

const app = express();
const PORT = process.env.PORT || 3001;
const NODE_ENV = process.env.NODE_ENV || 'development';

// 安全性標頭設定
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'"],
            fontSrc: ["'self'"],
            objectSrc: ["'none'"],
            mediaSrc: ["'self'"],
            frameSrc: ["'none'"],
        },
    },
    crossOriginEmbedderPolicy: false
}));

// CORS配置
const corsOptions = {
    origin: function (origin, callback) {
        const allowedOrigins = [
            'http://localhost:3000',
            'http://127.0.0.1:3000'
        ];
        
        // 開發環境允許無origin的請求 (如Postman)
        if (NODE_ENV === 'development' && !origin) return callback(null, true);
        
        if (allowedOrigins.indexOf(origin) !== -1) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
};

app.use(cors(corsOptions));

// 全域安全中間件
app.use(requestSizeLimit);
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(sanitizeInput);

// API 速率限制
app.use('/api/', apiLimiter);

// 資料庫初始化與連接池管理
const dbPath = path.join(__dirname, process.env.DB_PATH || 'db.sqlite3');
let db;

const initDatabase = () => {
    return new Promise((resolve, reject) => {
        db = new sqlite3.Database(dbPath, sqlite3.OPEN_READWRITE, async (err) => {
            if (err) {
                console.error('無法連接 SQLite:', err.message);
                reject(err);
            } else {
                console.log('已連接 SQLite 資料庫');
                
                // 啟用WAL模式提升效能
                db.run('PRAGMA journal_mode=WAL;');
                db.run('PRAGMA synchronous=NORMAL;');
                db.run('PRAGMA cache_size=10000;');
                db.run('PRAGMA temp_store=memory;');
                
                // 自動創建效能索引
                try {
                    await createPerformanceIndexes();
                    resolve();
                } catch (indexError) {
                    console.warn('⚠️ 索引創建警告:', indexError.message);
                    // 不要因為索引失敗而阻止服務啟動
                    resolve();
                }
            }
        });
    });
};

const createPerformanceIndexes = () => {
    return new Promise((resolve, reject) => {
        console.log('🔧 檢查並創建資料庫索引...');
        
        const indexes = [
            'CREATE INDEX IF NOT EXISTS idx_created_at ON semantic_clustering_sentiment(createdAt)',
            'CREATE INDEX IF NOT EXISTS idx_sentiment ON semantic_clustering_sentiment(sentiment)',
            'CREATE INDEX IF NOT EXISTS idx_cluster_id ON semantic_clustering_sentiment(cluster_id)',
            'CREATE INDEX IF NOT EXISTS idx_date_sentiment ON semantic_clustering_sentiment(createdAt, sentiment)',
            'CREATE INDEX IF NOT EXISTS idx_date_cluster ON semantic_clustering_sentiment(createdAt, cluster_id)',
            'CREATE INDEX IF NOT EXISTS idx_text_search ON semantic_clustering_sentiment(cleaned_text)'
        ];
        
        let completed = 0;
        const results = [];
        
        indexes.forEach((indexSql, index) => {
            const startTime = Date.now();
            db.run(indexSql, function(err) {
                const indexName = indexSql.match(/idx_\w+/)[0];
                
                if (err) {
                    if (err.message.includes('already exists')) {
                        console.log(`✅ 索引 ${indexName} 已存在`);
                        results.push({ index: indexName, status: 'exists' });
                    } else {
                        console.warn(`⚠️ 索引 ${indexName} 創建警告:`, err.message);
                        results.push({ index: indexName, status: 'warning', error: err.message });
                    }
                } else {
                    const time = Date.now() - startTime;
                    console.log(`✅ 索引 ${indexName} 創建完成 (${time}ms)`);
                    results.push({ index: indexName, status: 'created', time });
                }
                
                completed++;
                if (completed === indexes.length) {
                    // 更新統計信息
                    db.run('ANALYZE semantic_clustering_sentiment', (analyzeErr) => {
                        if (analyzeErr) {
                            console.warn('⚠️ 統計信息更新警告:', analyzeErr.message);
                        } else {
                            console.log('📊 表統計信息已更新');
                        }
                        
                        console.log('🎉 資料庫索引檢查完成');
                        resolve(results);
                    });
                }
            });
        });
    });
};

// 統一錯誤處理
const handleDatabaseError = (err, res) => {
    console.error('Database error:', err.message);
    res.status(500).json({
        error: 'Database operation failed',
        message: NODE_ENV === 'development' ? err.message : 'Internal server error'
    });
};

// 查詢包裝器，增加錯誤處理和日誌
const queryDatabase = (query, params = []) => {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();
        db.all(query, params, (err, rows) => {
            const duration = Date.now() - startTime;
            
            if (err) {
                console.error(`Query failed (${duration}ms):`, query, 'Params:', params, 'Error:', err.message);
                reject(err);
            } else {
                console.log(`Query executed (${duration}ms, ${rows.length} rows):`, query.substring(0, 100) + '...');
                resolve(rows);
            }
        });
    });
};

// K線 API
app.get('/api/kline', async (req, res) => {
    try {
        const rows = await queryDatabase('SELECT * FROM kline ORDER BY timestamp');
        res.json(rows);
    } catch (err) {
        handleDatabaseError(err, res);
    }
});

// UASTL 分解 API
app.get('/api/uastl', async (req, res) => {
    try {
        const rows = await queryDatabase('SELECT * FROM uastl ORDER BY date');
        res.json(rows);
    } catch (err) {
        handleDatabaseError(err, res);
    }
});

// Semantic Sentiment API (加強驗證)
app.get('/api/semantic', validateDateRange, async (req, res) => {
    try {
        const { startDate, endDate } = req.query;
        let query = 'SELECT createdAt, sentiment FROM semantic_clustering_sentiment';
        const params = [];

        if (startDate && endDate) {
            // 修正：處理ISO 8601格式的日期比較
            query += ' WHERE date(createdAt) BETWEEN ? AND ?';
            // 確保日期格式為 YYYY-MM-DD
            const start = new Date(startDate).toISOString().split('T')[0];
            const end = new Date(endDate).toISOString().split('T')[0];
            params.push(start, end);
        }
        query += ' ORDER BY createdAt';

        const rows = await queryDatabase(query, params);
        console.log(`Semantic query: ${query}, params: [${params.join(', ')}], results: ${rows.length}`);
        res.json(rows);
    } catch (err) {
        handleDatabaseError(err, res);
    }
});

// Term/N-gram Frequency API (加強驗證)
app.get('/api/term-ngram', validateDateRange, async (req, res) => {
    try {
        const { startDate, endDate } = req.query;
        let query = 'SELECT * FROM term_ngram_frequency';
        const params = [];

        if (startDate && endDate) {
            // 修正：確保日期格式一致
            query += ' WHERE date BETWEEN ? AND ?';
            const start = new Date(startDate).toISOString().split('T')[0];
            const end = new Date(endDate).toISOString().split('T')[0];
            params.push(start, end);
        }
        query += ' ORDER BY date, frequency DESC';

        const rows = await queryDatabase(query, params);
        console.log(`Term-ngram query: ${query}, params: [${params.join(', ')}], results: ${rows.length}`);
        res.json(rows);
    } catch (err) {
        handleDatabaseError(err, res);
    }
});

// Articles API (優化效能版本)
app.get('/api/articles', searchLimiter, validateArticleSearch, async (req, res) => {
    try {
        const { term, date, sentiment, page = 1, limit = 30 } = req.query;
        const offset = (page - 1) * limit;
        const maxLimit = Math.min(limit, 50); // 降低單次查詢限制

        console.log('Articles API called with params:', { term, date, sentiment, page, limit });
        
        let query = `
            SELECT id, cleaned_text, createdAt, sentiment 
            FROM semantic_clustering_sentiment
        `;
        const params = [];
        const conditions = [];

        // 優化查詢條件順序，最選擇性的條件在前
        if (date) {
            // 使用更精確的日期比較，處理ISO格式
            conditions.push('date(createdAt) = ?');
            // 確保日期格式為 YYYY-MM-DD，處理ISO日期
            const dateStr = new Date(date).toISOString().split('T')[0];
            params.push(dateStr);
            console.log('Date filter applied:', date, '->', dateStr);
        }
        if (sentiment) {
            conditions.push('sentiment = ?');
            params.push(sentiment);
            console.log('Sentiment filter applied:', sentiment);
        }
        if (term) {
            // 限制LIKE查詢的範圍
            if (term.length < 2) {
                return res.status(400).json({ 
                    error: 'Search term too short', 
                    message: '搜尋詞至少需要2個字元' 
                });
            }
            conditions.push('cleaned_text LIKE ?');
            params.push(`%${term}%`);
            console.log('Term filter applied:', term);
        }

        if (conditions.length > 0) {
            query += ' WHERE ' + conditions.join(' AND ');
        }

        // 如果沒有任何過濾條件，限制結果數量
        if (conditions.length === 0) {
            query += ' ORDER BY createdAt DESC LIMIT ? OFFSET ?';
            params.push(Math.min(maxLimit, 20), offset); // 無過濾時更嚴格的限制
        } else {
            query += ' ORDER BY createdAt DESC LIMIT ? OFFSET ?';
            params.push(maxLimit, offset);
        }

        console.log('Executing query:', query);
        console.log('With params:', params);
        
        const startTime = Date.now();
        const rows = await queryDatabase(query, params);
        const queryTime = Date.now() - startTime;
        
        console.log(`Articles query completed in ${queryTime}ms, returned ${rows.length} rows`);

        // 簡化總數計算 - 如果有結果且未達到限制，不需要額外計算
        let totalCount = rows.length;
        let totalPages = 1;
        
        if (rows.length === maxLimit) {
            // 只有在可能有更多結果時才執行count查詢
            let countQuery = query.replace(/SELECT.*?FROM/, 'SELECT COUNT(*) as count FROM')
                                 .replace(/ORDER BY.*?LIMIT.*?OFFSET.*?$/, '');
            const countParams = params.slice(0, -2);
            
            const countStartTime = Date.now();
            const countResult = await queryDatabase(countQuery, countParams);
            const countTime = Date.now() - countStartTime;
            
            totalCount = countResult[0]?.count || 0;
            totalPages = Math.ceil(totalCount / maxLimit);
            
            console.log(`Count query completed in ${countTime}ms, total: ${totalCount}`);
        }

        res.json({
            articles: rows,
            totalPages: totalPages,
            currentPage: parseInt(page),
            totalCount: totalCount,
            queryTime: queryTime
        });
    } catch (err) {
        console.error('Articles API error:', err);
        handleDatabaseError(err, res);
    }
});

// Clustering API (加強驗證)
app.get('/api/clusters', validateClusterQuery, async (req, res) => {
    try {
        const { startDate, endDate } = req.query;
        let query = 'SELECT x, y, cluster_id, cleaned_text FROM semantic_clustering_sentiment';
        const params = [];

        if (startDate && endDate) {
            // 修正：處理ISO 8601格式的日期比較
            query += ' WHERE date(createdAt) BETWEEN ? AND ?';
            const start = new Date(startDate).toISOString().split('T')[0];
            const end = new Date(endDate).toISOString().split('T')[0];
            params.push(start, end);
        }

        const rows = await queryDatabase(query, params);
        console.log(`Clusters query: ${query}, params: [${params.join(', ')}], results: ${rows.length}`);
        res.json(rows);
    } catch (err) {
        handleDatabaseError(err, res);
    }
});

// 資料庫索引優化端點 (僅開發環境使用)
app.post('/api/admin/create-indexes', async (req, res) => {
    if (process.env.NODE_ENV === 'production') {
        return res.status(403).json({ error: 'Not allowed in production' });
    }
    
    try {
        console.log('🔧 開始創建資料庫索引...');
        
        const indexes = [
            'CREATE INDEX IF NOT EXISTS idx_created_at ON semantic_clustering_sentiment(createdAt)',
            'CREATE INDEX IF NOT EXISTS idx_sentiment ON semantic_clustering_sentiment(sentiment)',
            'CREATE INDEX IF NOT EXISTS idx_cluster_id ON semantic_clustering_sentiment(cluster_id)',
            'CREATE INDEX IF NOT EXISTS idx_date_sentiment ON semantic_clustering_sentiment(createdAt, sentiment)',
            'CREATE INDEX IF NOT EXISTS idx_date_cluster ON semantic_clustering_sentiment(createdAt, cluster_id)',
            'CREATE INDEX IF NOT EXISTS idx_text_search ON semantic_clustering_sentiment(cleaned_text)'
        ];
        
        const results = [];
        
        for (const indexSql of indexes) {
            try {
                const startTime = Date.now();
                await new Promise((resolve, reject) => {
                    db.run(indexSql, (err) => {
                        if (err) reject(err);
                        else resolve();
                    });
                });
                const endTime = Date.now();
                
                const indexName = indexSql.match(/idx_\w+/)[0];
                results.push({
                    index: indexName,
                    status: 'success',
                    time: `${endTime - startTime}ms`
                });
                console.log(`✅ 索引 ${indexName} 創建完成 (${endTime - startTime}ms)`);
            } catch (error) {
                const indexName = indexSql.match(/idx_\w+/)[0];
                results.push({
                    index: indexName,
                    status: 'error',
                    error: error.message
                });
                console.error(`❌ 索引 ${indexName} 創建失敗:`, error.message);
            }
        }
        
        // 更新表統計信息
        try {
            console.log('📊 更新表統計信息...');
            await new Promise((resolve, reject) => {
                db.run('ANALYZE semantic_clustering_sentiment', (err) => {
                    if (err) reject(err);
                    else resolve();
                });
            });
            console.log('✅ 統計信息更新完成');
            results.push({ index: 'ANALYZE', status: 'success' });
        } catch (error) {
            console.error('❌ 統計信息更新失敗:', error.message);
            results.push({ index: 'ANALYZE', status: 'error', error: error.message });
        }
        
        // 檢查創建的索引
        const indexList = await new Promise((resolve, reject) => {
            db.all(`
                SELECT name, sql 
                FROM sqlite_master 
                WHERE type='index' 
                  AND tbl_name='semantic_clustering_sentiment'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            `, (err, rows) => {
                if (err) reject(err);
                else resolve(rows);
            });
        });
        
        console.log('🎉 索引優化完成!');
        console.log('現有索引:', indexList.map(idx => idx.name));
        
        res.json({
            message: '資料庫索引優化完成',
            results,
            indexes: indexList
        });
        
    } catch (error) {
        console.error('❌ 索引創建失敗:', error);
        res.status(500).json({ 
            error: '索引創建失敗', 
            details: error.message 
        });
    }
});

// 健康檢查端點
app.get('/health', (req, res) => {
    res.json({
        status: 'OK',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: NODE_ENV
    });
});

// 全域錯誤處理中間件
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: NODE_ENV === 'development' ? err.message : 'Something went wrong'
    });
});

// 404處理
app.use((req, res) => {
    res.status(404).json({
        error: 'Endpoint not found',
        message: `Cannot ${req.method} ${req.path}`
    });
});

// 正確的關閉處理
process.on('SIGINT', () => {
    console.log('Received SIGINT, shutting down gracefully...');
    if (db) {
        db.close((err) => {
            if (err) {
                console.error('Error closing database:', err.message);
            } else {
                console.log('Database connection closed.');
            }
            process.exit(0);
        });
    } else {
        process.exit(0);
    }
});

// 啟動伺服器
initDatabase()
    .then(() => {
        app.listen(PORT, () => {
            console.log(`🚀 API 伺服器啟動於 http://localhost:${PORT}`);
            console.log(`🛡️  環境: ${NODE_ENV}`);
            console.log(`📊 資料庫: ${dbPath}`);
        });
    })
    .catch((err) => {
        console.error('Failed to initialize server:', err);
        process.exit(1);
    }); 