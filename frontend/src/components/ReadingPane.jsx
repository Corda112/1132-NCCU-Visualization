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
    const [loadingMessage, setLoadingMessage] = useState(''); // 動態加載訊息

    useEffect(() => {
        const fetchArticles = async () => {
            setLoading(true);
            setError(null);
            setQueryTime(null);
            
            // 設置動態加載訊息
            if (filter?.sentiment && filter?.date) {
                const sentimentText = filter.sentiment === 'Positive' ? '正面' : 
                                    filter.sentiment === 'Negative' ? '負面' : '中性';
                setLoadingMessage(`正在載入 ${filter.date} 的${sentimentText}情緒推文...`);
            } else if (filter?.date) {
                setLoadingMessage(`正在載入 ${filter.date} 的所有推文...`);
            } else if (filter?.term) {
                setLoadingMessage(`正在搜尋「${filter.term}」相關推文...`);
            } else {
                setLoadingMessage('正在載入推文資料...');
            }
            
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
            setLoadingMessage('');
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
            <div style={{
                height: '400px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: '#fafafa',
                borderRadius: '8px',
                border: '1px solid #f0f0f0',
                margin: '10px 0'
            }}>
                <div style={{
                    width: '40px',
                    height: '40px',
                    border: '3px solid #f3f3f3',
                    borderTop: '3px solid #1890ff',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                    marginBottom: '16px'
                }}></div>
                <div style={{ color: '#666', fontSize: '16px', fontWeight: '500', marginBottom: '8px' }}>
                    {loadingMessage || '載入文章中...'}
                </div>
                {queryTime && (
                    <div style={{ fontSize: '12px', color: '#999' }}>
                        上次查詢耗時: {queryTime}ms
                    </div>
                )}
                <style jsx>{`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                `}</style>
            </div>
        );
    }

    if (error) {
        return (
            <div style={{
                padding: '20px',
                backgroundColor: '#fff2f0',
                border: '1px solid #ffccc7',
                borderRadius: '8px',
                margin: '10px 0'
            }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    marginBottom: '12px'
                }}>
                    <span style={{ color: '#ff4d4f', marginRight: '8px', fontSize: '18px' }}>⚠️</span>
                    <strong style={{ color: '#ff4d4f', fontSize: '16px' }}>載入失敗</strong>
                </div>
                <div style={{ marginBottom: '16px' }}>
                    <p style={{ margin: '6px 0', color: '#666', fontSize: '14px' }}>{error.message}</p>
                    {error.status && (
                        <p style={{ margin: '4px 0', fontSize: '12px', color: '#999' }}>
                            錯誤碼: {error.status}
                        </p>
                    )}
                </div>
                {error.isNetworkError && (
                    <button 
                        onClick={() => window.location.reload()}
                        style={{
                            padding: '8px 16px',
                            backgroundColor: '#1890ff',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '14px',
                            transition: 'background-color 0.3s'
                        }}
                        onMouseOver={(e) => e.target.style.backgroundColor = '#40a9ff'}
                        onMouseOut={(e) => e.target.style.backgroundColor = '#1890ff'}
                    >
                        重新載入
                    </button>
                )}
            </div>
        );
    }

    if (articles.length === 0 && !loading && !error) {
        const hasFilter = filter && (filter.term || filter.date || filter.sentiment);
        return (
            <div style={{
                height: '400px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: hasFilter ? '#fff7e6' : '#f6ffed',
                borderRadius: '8px',
                border: hasFilter ? '1px solid #ffd591' : '1px solid #d9f7be',
                margin: '10px 0',
                padding: '20px',
                textAlign: 'center'
            }}>
                {hasFilter ? (
                    <>
                        <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔍</div>
                        <p style={{ fontSize: '18px', fontWeight: '500', color: '#d46b08', marginBottom: '12px' }}>
                            未找到符合條件的文章
                        </p>
                        <div style={{ 
                            fontSize: '14px', 
                            color: '#8c8c8c', 
                            marginBottom: '20px',
                            backgroundColor: 'white',
                            padding: '12px',
                            borderRadius: '6px',
                            border: '1px solid #ffd591'
                        }}>
                            <div style={{ marginBottom: '4px', fontWeight: '500' }}>當前搜索條件：</div>
                            {filter.term && <div>搜索詞: "{filter.term}"</div>}
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
                                padding: '10px 20px',
                                backgroundColor: '#1890ff',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '14px',
                                fontWeight: '500',
                                transition: 'all 0.3s'
                            }}
                            onMouseOver={(e) => {
                                e.target.style.backgroundColor = '#40a9ff';
                                e.target.style.transform = 'translateY(-1px)';
                            }}
                            onMouseOut={(e) => {
                                e.target.style.backgroundColor = '#1890ff';
                                e.target.style.transform = 'translateY(0)';
                            }}
                        >
                            清除搜索條件
                        </button>
                    </>
                ) : (
                    <>
                        <div style={{ fontSize: '64px', marginBottom: '20px' }}>💡</div>
                        <p style={{ fontSize: '18px', fontWeight: '500', color: '#52c41a', marginBottom: '16px' }}>
                            請點擊左側圖表中的資料點來查看相關文章
                        </p>
                        <div style={{ 
                            fontSize: '14px', 
                            color: '#8c8c8c',
                            backgroundColor: 'white',
                            padding: '16px',
                            borderRadius: '8px',
                            border: '1px solid #d9f7be',
                            textAlign: 'left',
                            maxWidth: '400px'
                        }}>
                            <div style={{ fontWeight: '500', marginBottom: '8px', color: '#52c41a' }}>📊 可用操作：</div>
                            <div style={{ marginBottom: '4px' }}>• 點擊「社群情緒分析」圖表的線條</div>
                            <div style={{ marginBottom: '4px' }}>• 點擊「術語」或「N-gram」圖表的線條</div>
                            <div>• 點擊「聚類」圖表的散點</div>
                        </div>
                    </>
                )}
            </div>
        );
    }

    return (
        <div className="reading-pane" style={{ margin: '10px 0' }}>
            {/* 顯示當前搜索條件和效能資訊 */}
            {filter && (filter.term || filter.date || filter.sentiment) && (
                <div style={{ 
                    marginBottom: '20px', 
                    padding: '12px 16px', 
                    backgroundColor: '#e6f7ff', 
                    borderRadius: '8px',
                    fontSize: '14px',
                    border: '1px solid #bae7ff',
                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
                }}>
                    <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center',
                        marginBottom: '8px'
                    }}>
                        <div style={{ color: '#1890ff', fontWeight: '600' }}>
                            🔍 搜索結果
                        </div>
                        <div style={{ color: '#52c41a', fontWeight: '500' }}>
                            共 {articles.length} 篇文章
                        </div>
                    </div>
                    <div style={{ color: '#595959' }}>
                        {filter.term && <span style={{ 
                            backgroundColor: '#fff',
                            padding: '2px 8px',
                            borderRadius: '4px',
                            marginRight: '8px',
                            border: '1px solid #d9d9d9'
                        }}>詞彙「{filter.term}」</span>}
                        {filter.date && <span style={{ 
                            backgroundColor: '#fff',
                            padding: '2px 8px',
                            borderRadius: '4px',
                            marginRight: '8px',
                            border: '1px solid #d9d9d9'
                        }}>日期「{filter.date}」</span>}
                        {filter.sentiment && <span style={{ 
                            backgroundColor: filter.sentiment === 'Positive' ? '#f6ffed' : 
                                           filter.sentiment === 'Negative' ? '#fff2f0' : '#f0f5ff',
                            color: filter.sentiment === 'Positive' ? '#52c41a' : 
                                   filter.sentiment === 'Negative' ? '#ff4d4f' : '#1890ff',
                            padding: '2px 8px',
                            borderRadius: '4px',
                            marginRight: '8px',
                            border: `1px solid ${filter.sentiment === 'Positive' ? '#d9f7be' : 
                                                filter.sentiment === 'Negative' ? '#ffccc7' : '#bae7ff'}`
                        }}>情緒「{filter.sentiment}」</span>}
                    </div>
                    {queryTime && (
                        <div style={{ 
                            marginTop: '8px', 
                            color: '#8c8c8c', 
                            fontSize: '12px',
                            display: 'flex',
                            alignItems: 'center'
                        }}>
                            <span style={{ marginRight: '6px' }}>⚡</span>
                            查詢耗時: {queryTime}ms
                        </div>
                    )}
                </div>
            )}
            
            <div className="article-list">
                {articles.map((article, index) => (
                    <div key={article.id} className="article-item" style={{
                        backgroundColor: 'white',
                        border: '1px solid #f0f0f0',
                        borderRadius: '8px',
                        padding: '16px',
                        marginBottom: '12px',
                        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
                        transition: 'all 0.3s ease',
                        cursor: 'default'
                    }}
                    onMouseOver={(e) => {
                        e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.1)';
                        e.currentTarget.style.borderColor = '#d9d9d9';
                    }}
                    onMouseOut={(e) => {
                        e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.05)';
                        e.currentTarget.style.borderColor = '#f0f0f0';
                    }}
                    >
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start',
                            marginBottom: '8px'
                        }}>
                            <span style={{
                                backgroundColor: '#f6f6f6',
                                color: '#666',
                                padding: '2px 8px',
                                borderRadius: '12px',
                                fontSize: '12px',
                                fontWeight: '500'
                            }}>
                                #{index + 1}
                            </span>
                            <span style={{
                                color: '#8c8c8c',
                                fontSize: '12px'
                            }}>
                                {new Date(article.createdAt).toLocaleString('zh-TW', {
                                    year: 'numeric',
                                    month: '2-digit',
                                    day: '2-digit',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })}
                            </span>
                        </div>
                        <p style={{
                            margin: '0',
                            lineHeight: '1.6',
                            color: '#262626',
                            fontSize: '14px'
                        }}>
                            {article.cleaned_text}
                        </p>
                        {/* 添加情緒標籤 */}
                        {article.sentiment && (
                            <div style={{ marginTop: '12px' }}>
                                <span style={{
                                    backgroundColor: article.sentiment === 'Positive' ? '#f6ffed' : 
                                                   article.sentiment === 'Negative' ? '#fff2f0' : '#f0f5ff',
                                    color: article.sentiment === 'Positive' ? '#52c41a' : 
                                           article.sentiment === 'Negative' ? '#ff4d4f' : '#1890ff',
                                    padding: '4px 8px',
                                    borderRadius: '4px',
                                    fontSize: '12px',
                                    fontWeight: '500',
                                    border: `1px solid ${article.sentiment === 'Positive' ? '#d9f7be' : 
                                                        article.sentiment === 'Negative' ? '#ffccc7' : '#bae7ff'}`
                                }}>
                                    {article.sentiment === 'Positive' ? '😊 正面' : 
                                     article.sentiment === 'Negative' ? '😞 負面' : '😐 中性'}
                                </span>
                            </div>
                        )}
                    </div>
                ))}
            </div>
            
            {/* 分頁控制 */}
            {totalPages > 1 && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    marginTop: '20px',
                    gap: '8px'
                }}>
                    <button
                        onClick={() => handlePageChange(page - 1)}
                        disabled={page <= 1}
                        style={{
                            padding: '8px 12px',
                            border: '1px solid #d9d9d9',
                            backgroundColor: page <= 1 ? '#f5f5f5' : 'white',
                            color: page <= 1 ? '#bfbfbf' : '#595959',
                            borderRadius: '4px',
                            cursor: page <= 1 ? 'not-allowed' : 'pointer',
                            fontSize: '14px'
                        }}
                    >
                        上一頁
                    </button>
                    
                    <span style={{
                        padding: '8px 16px',
                        color: '#595959',
                        fontSize: '14px'
                    }}>
                        第 {page} 頁，共 {totalPages} 頁
                    </span>
                    
                    <button
                        onClick={() => handlePageChange(page + 1)}
                        disabled={page >= totalPages}
                        style={{
                            padding: '8px 12px',
                            border: '1px solid #d9d9d9',
                            backgroundColor: page >= totalPages ? '#f5f5f5' : 'white',
                            color: page >= totalPages ? '#bfbfbf' : '#595959',
                            borderRadius: '4px',
                            cursor: page >= totalPages ? 'not-allowed' : 'pointer',
                            fontSize: '14px'
                        }}
                    >
                        下一頁
                    </button>
                </div>
            )}
        </div>
    );
};

export default ReadingPane; 