const { body, query, validationResult } = require('express-validator');

// 錯誤處理中間件
const handleValidationErrors = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            error: 'Validation failed',
            details: errors.array()
        });
    }
    next();
};

// 日期格式驗證
const validateDate = (fieldName) => {
    return query(fieldName)
        .optional()
        .isISO8601({ strict: true })
        .withMessage(`${fieldName} must be a valid ISO 8601 date`)
        .toDate();
};

// 通用查詢參數驗證
const validatePagination = [
    query('page')
        .optional()
        .isInt({ min: 1 })
        .withMessage('Page must be a positive integer')
        .toInt(),
    query('limit')
        .optional()
        .isInt({ min: 1, max: 100 })
        .withMessage('Limit must be between 1 and 100')
        .toInt(),
    handleValidationErrors
];

// 文章搜尋驗證
const validateArticleSearch = [
    query('term')
        .optional()
        .isLength({ min: 1, max: 100 })
        .withMessage('Search term must be between 1 and 100 characters')
        .trim()
        .escape(),
    query('sentiment')
        .optional()
        .isIn(['Positive', 'Negative', 'Neutral'])
        .withMessage('Sentiment must be one of: Positive, Negative, Neutral'),
    validateDate('date'),
    ...validatePagination
];

// 時間範圍驗證
const validateDateRange = [
    validateDate('startDate'),
    validateDate('endDate'),
    (req, res, next) => {
        const { startDate, endDate } = req.query;
        if (startDate && endDate && new Date(startDate) > new Date(endDate)) {
            return res.status(400).json({
                error: 'startDate cannot be later than endDate'
            });
        }
        next();
    },
    handleValidationErrors
];

// 聚類查詢驗證
const validateClusterQuery = [
    ...validateDateRange
];

module.exports = {
    validateArticleSearch,
    validateDateRange,
    validateClusterQuery,
    validatePagination,
    handleValidationErrors
}; 