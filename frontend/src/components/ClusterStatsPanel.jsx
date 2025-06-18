import React, { useState, useEffect, useCallback } from 'react';
import { apiClient, API_ENDPOINTS, handleApiError } from '../config/api';

const ClusterStatsPanel = ({ range, selectedCluster, onClusterSelect, clusterStats, setClusterStats }) => {
    const [clusterDetail, setClusterDetail] = useState(null);
    const [loading, setLoading] = useState(false);
    const [detailLoading, setDetailLoading] = useState(false);
    const [error, setError] = useState(null);

    // 獲取聚類統計數據，只在沒有數據時才獲取
    const fetchClusterStats = useCallback(async () => {
        if (!range || !range.from || !range.to) return;
        if (clusterStats.length > 0) return; // 已有數據則跳過
        
        setLoading(true);
        setError(null);
        
        try {
            const startDate = new Date(range.from).toISOString().split('T')[0];
            const endDate = new Date(range.to).toISOString().split('T')[0];
            const response = await apiClient.get(API_ENDPOINTS.CLUSTER_STATS, {
                params: { startDate, endDate }
            });
            setClusterStats(response.data);
        } catch (error) {
            console.error('Error fetching cluster stats:', error);
            const errorInfo = handleApiError(error);
            setError(errorInfo);
        }
        setLoading(false);
    }, [range, clusterStats.length, setClusterStats]);

    // 獲取特定聚類的詳細信息
    const fetchClusterDetail = useCallback(async (clusterId) => {
        if (!clusterId || !range || !range.from || !range.to) return;
        
        setDetailLoading(true);
        
        try {
            const startDate = new Date(range.from).toISOString().split('T')[0];
            const endDate = new Date(range.to).toISOString().split('T')[0];
            const response = await apiClient.get(`${API_ENDPOINTS.CLUSTER_DETAIL}/${clusterId}`, {
                params: { startDate, endDate }
            });
            setClusterDetail(response.data);
        } catch (error) {
            console.error('Error fetching cluster detail:', error);
            setClusterDetail(null);
        }
        setDetailLoading(false);
    }, [range]);

    useEffect(() => {
        fetchClusterStats();
    }, [fetchClusterStats]);

    useEffect(() => {
        if (selectedCluster) {
            fetchClusterDetail(selectedCluster);
        } else {
            setClusterDetail(null);
        }
    }, [selectedCluster, fetchClusterDetail]);

    // 獲取情緒顏色
    const getSentimentColor = (sentiment) => {
        switch (sentiment) {
            case 'Positive': return '#52c41a';
            case 'Negative': return '#ff4d4f';
            case 'Neutral': return '#1890ff';
            default: return '#8c8c8c';
        }
    };

    // 獲取聚類大小描述
    const getClusterSizeLabel = (count, totalTweets) => {
        const percentage = (count / totalTweets * 100);
        if (percentage > 10) return '大型';
        if (percentage > 5) return '中型';
        if (percentage > 1) return '小型';
        return '微型';
    };

    if (loading) {
        return (
            <div style={{
                height: '400px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: '#fafafa',
                borderRadius: '8px'
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{
                        width: '32px',
                        height: '32px',
                        border: '3px solid #f3f3f3',
                        borderTop: '3px solid #1890ff',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite',
                        margin: '0 auto 12px'
                    }}></div>
                    <div style={{ color: '#666', fontSize: '14px' }}>載入聚類統計中...</div>
                </div>
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
                textAlign: 'center'
            }}>
                <div style={{ color: '#ff4d4f', marginBottom: '8px', fontSize: '16px' }}>⚠️</div>
                <div style={{ color: '#ff4d4f', fontSize: '14px' }}>載入聚類統計失敗</div>
                <div style={{ color: '#8c8c8c', fontSize: '12px', marginTop: '4px' }}>{error.message}</div>
            </div>
        );
    }

    const totalTweets = clusterStats.reduce((sum, cluster) => sum + cluster.tweet_count, 0);

    return (
        <div style={{ height: '400px', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {/* 頭部信息 */}
            <div style={{
                padding: '12px 16px',
                backgroundColor: '#f8f9fa',
                borderBottom: '1px solid #e9ecef',
                borderRadius: '8px 8px 0 0'
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h4 style={{ margin: 0, fontSize: '16px', color: '#262626' }}>聚類分析</h4>
                    <div style={{ fontSize: '12px', color: '#8c8c8c' }}>
                        {clusterStats.length} 個聚類 • {totalTweets.toLocaleString()} 篇推文
                    </div>
                </div>
            </div>

            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                {/* 左側：聚類列表 */}
                <div style={{ 
                    width: selectedCluster ? '40%' : '100%', 
                    borderRight: selectedCluster ? '1px solid #e9ecef' : 'none',
                    overflow: 'auto'
                }}>
                    {clusterStats.map((cluster, index) => {
                        const isSelected = selectedCluster === cluster.cluster_id;
                        return (
                            <div
                                key={cluster.cluster_id}
                                onClick={() => onClusterSelect?.(cluster.cluster_id)}
                                style={{
                                    padding: '12px 16px',
                                    borderBottom: '1px solid #f0f0f0',
                                    cursor: 'pointer',
                                    backgroundColor: isSelected ? '#e6f7ff' : 'white',
                                    transition: 'all 0.2s',
                                    borderLeft: isSelected ? '3px solid #1890ff' : '3px solid transparent'
                                }}
                                onMouseOver={(e) => {
                                    if (!isSelected) {
                                        e.currentTarget.style.backgroundColor = '#f8f9fa';
                                    }
                                }}
                                onMouseOut={(e) => {
                                    if (!isSelected) {
                                        e.currentTarget.style.backgroundColor = 'white';
                                    }
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                                            <span style={{ 
                                                fontWeight: '600', 
                                                fontSize: '14px',
                                                marginRight: '8px',
                                                color: '#262626' // 修正：明確設定深色字體
                                            }}>
                                                聚類 {cluster.cluster_id}
                                            </span>
                                            <span style={{
                                                backgroundColor: getSentimentColor(cluster.dominant_sentiment),
                                                color: 'white',
                                                padding: '2px 6px',
                                                borderRadius: '4px',
                                                fontSize: '10px',
                                                fontWeight: '500'
                                            }}>
                                                {cluster.dominant_sentiment}
                                            </span>
                                        </div>
                                        
                                        <div style={{ fontSize: '12px', color: '#595959', marginBottom: '6px' }}>
                                            {cluster.tweet_count.toLocaleString()} 篇推文 • 
                                            {getClusterSizeLabel(cluster.tweet_count, totalTweets)} • 
                                            密度 {cluster.density}%
                                        </div>

                                        {/* 情緒分佈條 */}
                                        <div style={{ 
                                            height: '4px', 
                                            backgroundColor: '#f0f0f0', 
                                            borderRadius: '2px',
                                            overflow: 'hidden',
                                            marginBottom: '6px'
                                        }}>
                                            <div style={{ 
                                                height: '100%', 
                                                display: 'flex'
                                            }}>
                                                <div style={{
                                                    width: `${cluster.sentiment_distribution.positive}%`,
                                                    backgroundColor: '#52c41a'
                                                }}></div>
                                                <div style={{
                                                    width: `${cluster.sentiment_distribution.negative}%`,
                                                    backgroundColor: '#ff4d4f'
                                                }}></div>
                                                <div style={{
                                                    width: `${cluster.sentiment_distribution.neutral}%`,
                                                    backgroundColor: '#1890ff'
                                                }}></div>
                                            </div>
                                        </div>

                                        {/* 代表性文本預覽 */}
                                        {cluster.representative_texts && cluster.representative_texts.length > 0 && (
                                            <div style={{ 
                                                fontSize: '11px', 
                                                color: '#595959',
                                                lineHeight: '1.4',
                                                fontStyle: 'italic'
                                            }}>
                                                "{cluster.representative_texts[0].text.substring(0, 50)}..."
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* 右側：聚類詳細信息 */}
                {selectedCluster && (
                    <div style={{ 
                        width: '60%', 
                        overflow: 'auto',
                        backgroundColor: '#fafafa'
                    }}>
                        {detailLoading ? (
                            <div style={{
                                height: '100%',
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center'
                            }}>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{
                                        width: '24px',
                                        height: '24px',
                                        border: '2px solid #f3f3f3',
                                        borderTop: '2px solid #1890ff',
                                        borderRadius: '50%',
                                        animation: 'spin 1s linear infinite',
                                        margin: '0 auto 8px'
                                    }}></div>
                                    <div style={{ color: '#666', fontSize: '12px' }}>載入詳細信息...</div>
                                </div>
                            </div>
                        ) : clusterDetail ? (
                            <div style={{ padding: '16px' }}>
                                {/* 聚類標題 */}
                                <div style={{
                                    marginBottom: '16px',
                                    paddingBottom: '12px',
                                    borderBottom: '1px solid #e9ecef'
                                }}>
                                    <h5 style={{ margin: '0 0 8px 0', fontSize: '16px' }}>
                                        聚類 {clusterDetail.cluster_id} 詳細分析
                                    </h5>
                                    <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#8c8c8c' }}>
                                        <span>總推文: {clusterDetail.stats.total_tweets}</span>
                                        <span>中心: ({clusterDetail.stats.center_x.toFixed(2)}, {clusterDetail.stats.center_y.toFixed(2)})</span>
                                    </div>
                                </div>

                                {/* 情緒分佈統計 */}
                                <div style={{ marginBottom: '16px' }}>
                                    <h6 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#262626' }}>情緒分佈</h6>
                                    <div style={{ display: 'flex', gap: '8px' }}>
                                        {[
                                            { label: '正面', value: clusterDetail.stats.sentiment_distribution.positive, color: '#52c41a' },
                                            { label: '負面', value: clusterDetail.stats.sentiment_distribution.negative, color: '#ff4d4f' },
                                            { label: '中性', value: clusterDetail.stats.sentiment_distribution.neutral, color: '#1890ff' }
                                        ].map(item => (
                                            <div key={item.label} style={{
                                                backgroundColor: 'white',
                                                padding: '8px 12px',
                                                borderRadius: '6px',
                                                textAlign: 'center',
                                                border: `1px solid ${item.color}20`,
                                                flex: 1
                                            }}>
                                                <div style={{ 
                                                    fontSize: '16px', 
                                                    fontWeight: '600', 
                                                    color: item.color 
                                                }}>
                                                    {item.value}%
                                                </div>
                                                <div style={{ fontSize: '11px', color: '#8c8c8c' }}>
                                                    {item.label}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* 代表性推文 */}
                                <div style={{ marginBottom: '16px' }}>
                                    <h6 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#262626' }}>代表性推文</h6>
                                    <div style={{ maxHeight: '200px', overflow: 'auto' }}>
                                        {clusterDetail.tweets.slice(0, 5).map((tweet, index) => (
                                            <div key={index} style={{
                                                backgroundColor: 'white',
                                                padding: '8px 12px',
                                                marginBottom: '6px',
                                                borderRadius: '6px',
                                                border: '1px solid #f0f0f0'
                                            }}>
                                                <div style={{ 
                                                    fontSize: '12px', 
                                                    lineHeight: '1.4',
                                                    marginBottom: '4px'
                                                }}>
                                                    {tweet.text.substring(0, 120)}...
                                                </div>
                                                <div style={{ 
                                                    display: 'flex', 
                                                    justifyContent: 'space-between',
                                                    fontSize: '10px',
                                                    color: '#8c8c8c'
                                                }}>
                                                    <span style={{
                                                        color: getSentimentColor(tweet.sentiment),
                                                        fontWeight: '500'
                                                    }}>
                                                        {tweet.sentiment}
                                                    </span>
                                                    <span>{new Date(tweet.date).toLocaleDateString()}</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div style={{
                                height: '100%',
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center',
                                color: '#8c8c8c'
                            }}>
                                無法載入聚類詳細信息
                            </div>
                        )}
                    </div>
                )}
            </div>

            <style jsx>{`
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
};

export default ClusterStatsPanel; 