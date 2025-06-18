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
        db = new sqlite3.Database(dbPath, sqlite3.OPEN_READWRITE, (err) => {
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
                
                resolve();
            }
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

// Articles API (加強安全性和驗證)
app.get('/api/articles', searchLimiter, validateArticleSearch, async (req, res) => {
    try {
        const { term, date, sentiment, page = 1, limit = 30 } = req.query;
        const offset = (page - 1) * limit;
        const maxLimit = Math.min(limit, 100); // 強制限制最大值

        let query = `
            SELECT id, cleaned_text, createdAt, sentiment 
            FROM semantic_clustering_sentiment
        `;
        const params = [];
        const conditions = [];

        if (term) {
            conditions.push('cleaned_text LIKE ?');
            params.push(`%${term}%`);
        }
        if (date) {
            conditions.push('date(createdAt) = date(?)');
            params.push(date);
        }
        if (sentiment) {
            conditions.push('sentiment = ?');
            params.push(sentiment);
        }

        if (conditions.length > 0) {
            query += ' WHERE ' + conditions.join(' AND ');
        }

        // 分頁查詢
        query += ` ORDER BY createdAt DESC LIMIT ? OFFSET ?`;
        params.push(maxLimit, offset);

        const rows = await queryDatabase(query, params);

        // 計算總數 (優化版本)
        let countQuery = query.replace(/SELECT.*?FROM/, 'SELECT COUNT(*) as count FROM')
                             .replace(/ORDER BY.*?LIMIT.*?OFFSET.*?$/, '');
        const countParams = params.slice(0, -2);

        const countResult = await queryDatabase(countQuery, countParams);
        const totalCount = countResult[0]?.count || 0;

        res.json({
            articles: rows,
            totalPages: Math.ceil(totalCount / maxLimit),
            currentPage: page,
            totalCount: totalCount
        });
    } catch (err) {
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