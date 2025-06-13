const express = require('express');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// 資料庫初始化
const dbPath = path.join(__dirname, 'db.sqlite3');
const db = new sqlite3.Database(dbPath, (err) => {
    if (err) {
        console.error('無法連接 SQLite:', err.message);
    } else {
        console.log('已連接 SQLite 資料庫');
    }
});

// K線 API
app.get('/api/kline', (req, res) => {
    db.all('SELECT * FROM kline ORDER BY timestamp', [], (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else {
            res.json(rows);
        }
    });
});

// UASTL 分解 API
app.get('/api/uastl', (req, res) => {
    db.all('SELECT * FROM uastl ORDER BY date', [], (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else {
            res.json(rows);
        }
    });
});

// Semantic Sentiment API
app.get('/api/semantic', (req, res) => {
    const { startDate, endDate } = req.query;
    let query = 'SELECT createdAt, sentiment FROM semantic_clustering_sentiment';
    const params = [];

    if (startDate && endDate) {
        query += ' WHERE date(createdAt) BETWEEN date(?) AND date(?)';
        params.push(startDate, endDate);
    }
    query += ' ORDER BY createdAt';

    db.all(query, params, (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else {
            res.json(rows);
        }
    });
});

// Term/N-gram Frequency API
app.get('/api/term-ngram', (req, res) => {
    const { startDate, endDate } = req.query;
    let query = 'SELECT * FROM term_ngram_frequency';
    const params = [];

    if (startDate && endDate) {
        query += ' WHERE date(date) BETWEEN date(?) AND date(?)';
        params.push(startDate, endDate);
    }
    query += ' ORDER BY date, frequency DESC';

    db.all(query, params, (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else {
            res.json(rows);
        }
    });
});

// Articles for Reading Pane API
app.get('/api/articles', (req, res) => {
    const { term, date, sentiment, page = 1, limit = 30 } = req.query;
    const offset = (page - 1) * limit;

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

    // Add ordering and pagination
    query += ` ORDER BY createdAt DESC LIMIT ? OFFSET ?`;
    params.push(limit, offset);

    db.all(query, params, (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
            return;
        }

        // Also get total count for pagination
        let countQuery = query.replace(/SELECT .*? FROM/, 'SELECT COUNT(*) as count FROM').replace(/LIMIT \? OFFSET \?/, '');
        const countParams = params.slice(0, -2); // Remove limit and offset

        db.get(countQuery, countParams, (err, countRow) => {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json({
                    articles: rows,
                    totalPages: Math.ceil((countRow.count || 0) / limit)
                });
            }
        });
    });
});

// Clustering API
app.get('/api/clusters', (req, res) => {
    const { startDate, endDate } = req.query;
    let query = 'SELECT x, y, cluster_id, cleaned_text FROM semantic_clustering_sentiment';
    const params = [];

    if (startDate && endDate) {
        query += ' WHERE date(createdAt) BETWEEN date(?) AND date(?)';
        params.push(startDate, endDate);
    }

    db.all(query, params, (err, rows) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else {
            res.json(rows);
        }
    });
});

app.listen(PORT, () => {
    console.log(`API 伺服器啟動於 http://localhost:${PORT}`);
}); 