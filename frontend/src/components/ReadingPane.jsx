import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { getApiUrl, API_ENDPOINTS, handleApiError } from '../config/api';
import './ReadingPane.css';

const ReadingPane = ({ filter }) => {
    const [articles, setArticles] = useState([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchArticles = async () => {
            setLoading(true);
            setError(null);
            
            try {
                const apiUrl = getApiUrl(API_ENDPOINTS.ARTICLES);
                const requestParams = { ...filter, page };
                
                console.log('ReadingPane: Making API request to:', apiUrl);
                console.log('ReadingPane: Request params:', requestParams);
                
                const response = await axios.get(apiUrl, {
                    params: requestParams,
                    timeout: 10000
                });
                
                console.log('ReadingPane: API response:', response.data);
                
                setArticles(response.data.articles || []);
                setTotalPages(response.data.totalPages || 0);
            } catch (error) {
                console.error('ReadingPane: API error:', error);
                const errorInfo = handleApiError(error);
                setError(errorInfo);
                
                // 如果是速率限制，顯示特殊訊息
                if (errorInfo.isRateLimited) {
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
        
        console.log('ReadingPane: Current filter:', filter);
        
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
        return <div className="loading-pane">載入文章中...</div>;
    }

    if (error) {
        return (
            <div className="error-pane">
                <div className="error-message">
                    <h4>載入失敗</h4>
                    <p>{error.message}</p>
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
                    </div>
                ) : (
                    '點擊「詞彙」或「N-gram」圖表中的資料點來查看相關文章'
                )}
            </div>
        );
    }

    return (
        <div className="reading-pane">
            {/* 顯示當前搜索條件 */}
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