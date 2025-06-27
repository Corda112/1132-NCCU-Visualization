import axios from 'axios';

// API 配置
const API_BASE_URL = process.env.REACT_APP_API_URL || 
    (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:3001');

// API 端點
export const API_ENDPOINTS = {
    KLINE: '/api/kline',
    UASTL: '/api/uastl',
    SEMANTIC: '/api/semantic',
    TERM_NGRAM: '/api/term-ngram',
    ARTICLES: '/api/articles',
    CLUSTERS: '/api/clusters'
};

// 獲取完整的 API URL
export const getApiUrl = (endpoint) => {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log('API URL:', url);
    return url;
};

// API 錯誤處理函數
export const handleApiError = (error) => {
    console.error('API Error:', error);
    
    if (!error.response) {
        // 網絡錯誤
        return {
            message: '網絡連接失敗，請檢查網絡連接或稍後重試',
            isNetworkError: true,
            status: 0
        };
    }

    const status = error.response.status;
    const data = error.response.data;

    if (status === 429) {
        return {
            message: '請求過於頻繁，請稍後再試',
            isRateLimited: true,
            status: status
        };
    }

    if (status === 400) {
        return {
            message: data?.message || '請求參數錯誤',
            status: status
        };
    }

    if (status === 500) {
        return {
            message: '伺服器內部錯誤，查詢時間可能過長',
            status: status
        };
    }

    return {
        message: data?.message || `請求失敗 (${status})`,
        status: status
    };
};

// 創建帶有預設配置的 axios 實例
export const apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 增加到30秒
    headers: {
        'Content-Type': 'application/json'
    }
});

// 請求攔截器
apiClient.interceptors.request.use(
    (config) => {
        console.log('API Request:', config.method?.toUpperCase(), config.url, config.params);
        return config;
    },
    (error) => {
        console.error('Request Error:', error);
        return Promise.reject(error);
    }
);

// 響應攔截器
apiClient.interceptors.response.use(
    (response) => {
        console.log('API Response:', response.status, response.config.url, 
                   'Data length:', Array.isArray(response.data) ? response.data.length : 'Object');
        return response;
    },
    (error) => {
        console.error('Response Error:', error.response?.status, error.config?.url, error.message);
        return Promise.reject(error);
    }
); 