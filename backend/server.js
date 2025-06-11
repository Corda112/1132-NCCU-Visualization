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

app.listen(PORT, () => {
    console.log(`API 伺服器啟動於 http://localhost:${PORT}`);
}); 