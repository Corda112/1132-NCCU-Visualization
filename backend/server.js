const express = require('express');
const cors = require('cors');
const fs = require('fs');
let sqlite3;
try {
    sqlite3 = require('sqlite3').verbose();
} catch (e) {
    console.warn('sqlite3 module not available, falling back to JSON data');
}
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// 資料庫初始化（若 sqlite3 不可用則使用 JSON 檔案）
let db = null;
if (sqlite3) {
    const dbPath = path.join(__dirname, 'db.sqlite3');
    db = new sqlite3.Database(dbPath, (err) => {
        if (err) {
            console.error('無法連接 SQLite:', err.message);
        } else {
            console.log('已連接 SQLite 資料庫');
        }
    });
}

// 載入備用 JSON 資料
const semanticJSON = JSON.parse(fs.readFileSync(path.join(__dirname, 'Final_semantic_clustering_sentiment.json')));
const termJSON = JSON.parse(fs.readFileSync(path.join(__dirname, 'Final_term_ngram_frequency.json')));
const klineJSON = fs.existsSync(path.join(__dirname, 'kline_sample.json')) ?
    JSON.parse(fs.readFileSync(path.join(__dirname, 'kline_sample.json'))) : [];
const uastlJSON = fs.existsSync(path.join(__dirname, 'uastl_sample.json')) ?
    JSON.parse(fs.readFileSync(path.join(__dirname, 'uastl_sample.json'))) : [];

// K線 API
app.get('/api/kline', (req, res) => {
    if (db) {
        db.all('SELECT * FROM kline ORDER BY timestamp', [], (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json(rows);
            }
        });
    } else {
        res.json(klineJSON);
    }
});

// UASTL 分解 API
app.get('/api/uastl', (req, res) => {
    if (db) {
        db.all('SELECT * FROM uastl ORDER BY date', [], (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json(rows);
            }
        });
    } else {
        res.json(uastlJSON);
    }
});

// Semantic Sentiment API
app.get('/api/semantic', (req, res) => {
    const { startDate, endDate } = req.query;
    if (db) {
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
    } else {
        let rows = semanticJSON.map(d => ({ createdAt: d.createdAt, sentiment: d.sentiment }));
        if (startDate && endDate) {
            rows = rows.filter(r => {
                const d = r.createdAt.split('T')[0];
                return d >= startDate && d <= endDate;
            });
        }
        rows.sort((a,b) => new Date(a.createdAt) - new Date(b.createdAt));
        res.json(rows);
    }
});

// Term/N-gram Frequency API
app.get('/api/term-ngram', (req, res) => {
    const { startDate, endDate } = req.query;
    if (db) {
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
    } else {
        let rows = [];
        for (const date in termJSON) {
            if (!termJSON.hasOwnProperty(date)) continue;
            if (startDate && endDate) {
                if (date < startDate || date > endDate) continue;
            }
            const terms = termJSON[date];
            for (const term in terms) {
                rows.push({ date, term, frequency: terms[term] });
            }
        }
        rows.sort((a,b) => new Date(a.date) - new Date(b.date) || b.frequency - a.frequency);
        res.json(rows);
    }
});

// Articles for Reading Pane API
app.get('/api/articles', (req, res) => {
    const { term, date, sentiment, page = 1, limit = 30 } = req.query;
    const offset = (page - 1) * limit;

    if (db) {
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
    } else {
        let results = semanticJSON;
        if (term) {
            const t = term.toLowerCase();
            results = results.filter(d => d.cleaned_text.toLowerCase().includes(t));
        }
        if (date) {
            results = results.filter(d => d.createdAt.startsWith(date));
        }
        if (sentiment) {
            results = results.filter(d => d.sentiment === sentiment);
        }
        const totalPages = Math.ceil(results.length / limit);
        const articles = results.slice(offset, offset + Number(limit)).map(d => ({
            id: d.id,
            cleaned_text: d.cleaned_text,
            createdAt: d.createdAt,
            sentiment: d.sentiment
        }));
        res.json({ articles, totalPages });
    }
});

// Clustering API
app.get('/api/clusters', (req, res) => {
    const { startDate, endDate } = req.query;
    if (db) {
        let query = 'SELECT x, y, cluster_id, cleaned_text, sentiment, createdAt FROM semantic_clustering_sentiment';
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
    } else {
        let rows = semanticJSON.map(d => ({
            x: d.x,
            y: d.y,
            cluster_id: d.cluster_id,
            cleaned_text: d.cleaned_text,
            sentiment: d.sentiment,
            createdAt: d.createdAt
        }));
        if (startDate && endDate) {
            rows = rows.filter(r => {
                const d = r.createdAt.split('T')[0];
                return d >= startDate && d <= endDate;
            });
        }
        res.json(rows);
    }
});

app.listen(PORT, () => {
    console.log(`API 伺服器啟動於 http://localhost:${PORT}`);
}); 