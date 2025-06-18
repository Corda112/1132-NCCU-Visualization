// API 配置
const API_CONFIG = {
    BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:3001',
    TIMEOUT: 10000, // 10秒超時
    RETRY_ATTEMPTS: 3
};

// API 端點
export const API_ENDPOINTS = {
    KLINE: '/api/kline',
    UASTL: '/api/uastl',
    SEMANTIC: '/api/semantic',
    TERM_NGRAM: '/api/term-ngram',
    ARTICLES: '/api/articles',
    CLUSTERS: '/api/clusters',
    HEALTH: '/health'
};

// 請求配置
export const getApiUrl = (endpoint) => {
    return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// 錯誤處理
export const handleApiError = (error) => {
    if (error.response) {
        // 伺服器回應錯誤
        const status = error.response.status;
        const message = error.response.data?.message || 'Unknown error';
        
        switch (status) {
            case 400:
                console.error('請求參數錯誤:', message);
                break;
            case 429:
                console.error('請求過於頻繁，請稍後再試');
                break;
            case 500:
                console.error('伺服器內部錯誤:', message);
                break;
            default:
                console.error(`API錯誤 ${status}:`, message);
        }
        
        return {
            status,
            message,
            isRateLimited: status === 429
        };
    } else if (error.request) {
        // 網路錯誤
        console.error('網路連接錯誤:', error.message);
        return {
            status: 0,
            message: '無法連接到伺服器，請檢查網路連接',
            isNetworkError: true
        };
    } else {
        // 其他錯誤
        console.error('未知錯誤:', error.message);
        return {
            status: -1,
            message: '發生未知錯誤',
            isUnknownError: true
        };
    }
};

export default API_CONFIG; 