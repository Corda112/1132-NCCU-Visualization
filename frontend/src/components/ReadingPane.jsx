import React, { useState, useEffect } from 'react';
import { apiClient, API_ENDPOINTS, handleApiError } from '../config/api';
import './ReadingPane.css';

const ReadingPane = ({ filter }) => {
    const [articles, setArticles] = useState([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [queryTime, setQueryTime] = useState(null);

    useEffect(() => {
        const fetchArticles = async () => {
            setLoading(true);
            setError(null);
            setQueryTime(null);
            
            try {
                const requestParams = { ...filter, page };
                
                console.log('ReadingPane: Making API request');
                console.log('ReadingPane: Request params:', requestParams);
                console.log('ReadingPane: Current filter state:', filter);
                
                const startTime = Date.now();
                const response = await apiClient.get(API_ENDPOINTS.ARTICLES, {
                    params: requestParams
                });
                const requestTime = Date.now() - startTime;
                
                console.log('ReadingPane: API response received');
                console.log('ReadingPane: Articles count:', response.data.articles?.length || 0);
                console.log('ReadingPane: Query time from server:', response.data.queryTime, 'ms');
                console.log('ReadingPane: Total request time:', requestTime, 'ms');
                
                setArticles(response.data.articles || []);
                setTotalPages(response.data.totalPages || 0);
                setQueryTime(response.data.queryTime || requestTime);
            } catch (error) {
                console.error('ReadingPane: API error:', error);
                console.error('ReadingPane: Error details:', {
                    message: error.message,
                    response: error.response?.data,
                    status: error.response?.status
                });
                const errorInfo = handleApiError(error);
                setError(errorInfo);
                
                // 特殊錯誤處理
                if (error.code === 'ECONNABORTED') {
                    setError({
                        ...errorInfo,
                        message: '查詢超時，資料量過大，請縮小搜尋範圍'
                    });
                } else if (errorInfo.isRateLimited) {
                    setError({
                        ...errorInfo,
                        message: '搜尋過於頻繁，請等待5分鐘後再試'
                    });
                }
            }
            setLoading(false);
        };

        // 輸入驗證
        if (filter && filter.term && filter.term.length > 100) {
            setError({
                status: 400,
                message: '搜尋詞彙過長，請限制在100字元以內'
            });
            return;
        }
        
        console.log('ReadingPane: useEffect triggered, current filter:', filter);
        console.log('ReadingPane: Filter object keys:', Object.keys(filter || {}));
        
        // 不管是否有filter，都嘗試載入文章
        fetchArticles();

    }, [filter, page]);

    // Reset page to 1 when filter changes
    useEffect(() => {
        setPage(1);
    }, [filter]);

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= totalPages) {
            setPage(newPage);
        }
    };

    if (loading) {
        return (
            <div className="loading-pane">
                載入文章中...
                {queryTime && <div style={{ fontSize: '0.8em', color: '#666' }}>
                    上次查詢耗時: {queryTime}ms
                </div>}
            </div>
        );
    }

    if (error) {
        return (
            <div className="error-pane">
                <div className="error-message">
                    <h4>載入失敗</h4>
                    <p>{error.message}</p>
                    {error.status && <p style={{ fontSize: '0.8em', color: '#666' }}>
                        錯誤碼: {error.status}
                    </p>}
                    {error.isNetworkError && (
                        <button onClick={() => window.location.reload()}>
                            重新載入
                        </button>
                    )}
                </div>
            </div>
        );
    }

    if (articles.length === 0 && !loading && !error) {
        const hasFilter = filter && (filter.term || filter.date || filter.sentiment);
        return (
            <div className="info-pane">
                {hasFilter ? (
                    <div>
                        <p>未找到符合條件的文章</p>
                        <div style={{ fontSize: '0.9em', color: '#666', marginTop: '8px' }}>
                            {filter.term && <div>搜索詞: {filter.term}</div>}
                            {filter.date && <div>日期: {filter.date}</div>}
                            {filter.sentiment && <div>情緒: {filter.sentiment}</div>}
                        </div>
                        <button 
                            onClick={() => {
                                console.log('ReadingPane: Clearing filter');
                                setArticles([]);
                                // 這裡需要通知父組件清除過濾器
                            }}
                            style={{ 
                                marginTop: '10px', 
                                padding: '5px 10px',
                                backgroundColor: '#f0f0f0',
                                border: '1px solid #ccc',
                                borderRadius: '4px',
                                cursor: 'pointer'
                            }}
                        >
                            清除搜索條件
                        </button>
                    </div>
                ) : (
                    <div>
                        <p>💡 請點擊左側圖表中的資料點來查看相關文章</p>
                        <div style={{ fontSize: '0.85em', color: '#888', marginTop: '8px' }}>
                            <p>• 點擊「社群情緒分析」圖表的線條</p>
                            <p>• 點擊「術語」或「N-gram」圖表的線條</p>
                            <p>• 點擊「聚類」圖表的散點</p>
                        </div>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="reading-pane">
            {/* 顯示當前搜索條件和效能資訊 */}
            {filter && (filter.term || filter.date || filter.sentiment) && (
                <div className="search-info" style={{ 
                    marginBottom: '15px', 
                    padding: '8px 12px', 
                    backgroundColor: '#f0f8ff', 
                    borderRadius: '6px',
                    fontSize: '0.9em',
                    color: '#2c5aa0'
                }}>
                    <strong>搜索條件:</strong>
                    {filter.term && <span> 詞彙「{filter.term}」</span>}
                    {filter.date && <span> 日期「{filter.date}」</span>}
                    {filter.sentiment && <span> 情緒「{filter.sentiment}」</span>}
                    <span style={{ marginLeft: '8px', color: '#666' }}>
                        (共 {articles.length} 篇文章)
                    </span>
                    {queryTime && (
                        <span style={{ marginLeft: '8px', color: '#999', fontSize: '0.8em' }}>
                            查詢耗時: {queryTime}ms
                        </span>
                    )}
                </div>
            )}
            
            <div className="article-list">
                {articles.map(article => (
                    <div key={article.id} className="article-item">
                        <p className="article-text">{article.cleaned_text}</p>
                        <div className="article-meta">
                            <span>{new Date(article.createdAt).toLocaleString()}</span>
                            <span className={`sentiment-tag ${article.sentiment}`}>
                                {article.sentiment}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
            
            {totalPages > 1 && (
                <div className="pagination">
                    <button 
                        onClick={() => handlePageChange(page - 1)} 
                        disabled={page <= 1}
                    >
                        上一頁
                    </button>
                    <span>第 {page} 頁，共 {totalPages} 頁</span>
                    <button 
                        onClick={() => handlePageChange(page + 1)} 
                        disabled={page >= totalPages}
                    >
                        下一頁
                    </button>
                </div>
            )}
        </div>
    );
};

export default ReadingPane; 