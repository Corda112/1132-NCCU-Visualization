const rateLimit = require('express-rate-limit');

// 基本API速率限制
const apiLimiter = rateLimit({
    windowMs: process.env.RATE_LIMIT_WINDOW_MS || 15 * 60 * 1000, // 15分鐘
    max: process.env.RATE_LIMIT_MAX_REQUESTS || 100, // 每個IP最多100次請求
    message: {
        error: 'Too many requests from this IP, please try again later.',
        retryAfter: Math.ceil((process.env.RATE_LIMIT_WINDOW_MS || 15 * 60 * 1000) / 1000)
    },
    standardHeaders: true,
    legacyHeaders: false,
    // 針對搜尋API的嚴格限制
    skip: (req) => {
        // 跳過靜態資源
        return req.path.includes('static') || req.path.includes('assets');
    }
});

// 搜尋API特殊限制 (更嚴格)
const searchLimiter = rateLimit({
    windowMs: 5 * 60 * 1000, // 5分鐘
    max: 20, // 每個IP最多20次搜尋請求
    message: {
        error: 'Too many search requests, please try again later.',
        retryAfter: 300
    },
    standardHeaders: true,
    legacyHeaders: false
});

// SQL注入防護
const sanitizeInput = (req, res, next) => {
    // 檢查危險關鍵字
    const dangerousPatterns = [
        /(\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|EXEC|EXECUTE)\b)/i,
        /(UNION.*SELECT)/i,
        /(SELECT.*FROM)/i,
        /('|(\\')|(\"|(\\\")))(\s)*((\d+)|(\w+))/i,
        /(OR|AND)\s+\d+\s*=\s*\d+/i
    ];

    const checkValue = (value) => {
        if (typeof value === 'string') {
            return dangerousPatterns.some(pattern => pattern.test(value));
        }
        return false;
    };

    // 檢查查詢參數
    for (const [key, value] of Object.entries(req.query)) {
        if (checkValue(value)) {
            return res.status(400).json({
                error: 'Invalid input detected',
                message: 'Request contains potentially harmful content'
            });
        }
    }

    // 檢查body參數
    if (req.body) {
        for (const [key, value] of Object.entries(req.body)) {
            if (checkValue(value)) {
                return res.status(400).json({
                    error: 'Invalid input detected',
                    message: 'Request contains potentially harmful content'
                });
            }
        }
    }

    next();
};

// 請求大小限制
const requestSizeLimit = (req, res, next) => {
    const contentLength = req.headers['content-length'];
    const maxSize = 1024 * 1024; // 1MB

    if (contentLength && parseInt(contentLength) > maxSize) {
        return res.status(413).json({
            error: 'Request too large',
            message: 'Request size exceeds maximum allowed limit'
        });
    }

    next();
};

module.exports = {
    apiLimiter,
    searchLimiter,
    sanitizeInput,
    requestSizeLimit
}; 