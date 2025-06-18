import React, { useState, useEffect, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { apiClient, API_ENDPOINTS, handleApiError } from '../config/api';

const FrequencyChart = ({ range, type, onTermSelect }) => {
    const [chartData, setChartData] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const maxRetries = 3;

    const fetchData = useCallback(async (isRetry = false) => {
        if (!range || !range.from || !range.to) return;
        
        setLoading(true);
        if (!isRetry) {
            setError(null);
            setRetryCount(0);
        }
        
        try {
            const startDate = new Date(range.from).toISOString().split('T')[0];
            const endDate = new Date(range.to).toISOString().split('T')[0];
            const response = await apiClient.get(API_ENDPOINTS.TERM_NGRAM, {
                params: { startDate, endDate }
            });

            // Filter and process data based on type (term or ngram)
            const isNgram = (term) => term.includes(' ');
            const filteredData = response.data.filter(item => {
                return type === 'ngram' ? isNgram(item.term) : !isNgram(item.term);
            });

            // Get top 10 most frequent terms/ngrams for the period
            const frequencyMap = filteredData.reduce((acc, { term, frequency }) => {
                acc[term] = (acc[term] || 0) + frequency;
                return acc;
            }, {});

            const top10 = Object.entries(frequencyMap)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 10)
                .map(([term]) => term);

            // Group data by date for the top 10 terms
            const dailyData = filteredData.reduce((acc, { date, term, frequency }) => {
                const day = new Date(date).toISOString().split('T')[0];
                if (top10.includes(term)) {
                    if (!acc[day]) acc[day] = { date: day };
                    acc[day][term] = frequency;
                }
                return acc;
            }, {});

            setChartData({
                dates: Object.keys(dailyData).sort(),
                series: top10.map(term => ({
                    name: term,
                    type: 'line',
                    data: Object.values(dailyData).sort((a, b) => new Date(a.date) - new Date(b.date)).map(d => d[term] || 0)
                })),
                top10terms: top10
            });

            setError(null);
            setRetryCount(0);
        } catch (error) {
            console.error(`Error fetching ${type} data:`, error);
            const errorInfo = handleApiError(error);
            
            // 錯誤統計和重試邏輯
            if (errorInfo.isNetworkError && retryCount < maxRetries) {
                setRetryCount(prev => prev + 1);
                setTimeout(() => {
                    fetchData(true);
                }, 1000 * Math.pow(2, retryCount));
            } else {
                setError({
                    ...errorInfo,
                    retryCount,
                    canRetry: retryCount < maxRetries,
                    component: `FrequencyChart-${type}`
                });
            }
        }
        setLoading(false);
    }, [range, type, retryCount]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    const handleRetry = () => {
        if (retryCount < maxRetries) {
            fetchData(true);
        }
    };

    const getOption = () => ({
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                return `${params[0].axisValue}<br/>${params.map(p => 
                    `${p.marker}${p.seriesName}: ${p.value}`
                ).join('<br/>')}`;
            }
        },
        legend: {
            data: chartData.top10terms,
            orient: 'vertical',
            align: 'left',
            left: 'right',
            type: 'scroll'
        },
        xAxis: {
            type: 'category',
            data: chartData.dates
        },
        yAxis: {
            type: 'value'
        },
        series: chartData.series?.map(s => ({
            ...s,
            smooth: true,
            lineStyle: { width: 2 }
        })) || []
    });

    const onChartClick = (params) => {
        console.log('FrequencyChart: Chart clicked!', params);
        if (onTermSelect) {
            const term = params.seriesName;
            const date = params.name;
            console.log('FrequencyChart: Sending term and date:', { term, date });
            onTermSelect(term, date);
        }
    };

    if (loading) {
        return (
            <div className="loading-pane">
                <div>載入{type === 'ngram' ? 'N-gram' : '詞頻'}資料中...</div>
                {retryCount > 0 && (
                    <div style={{ fontSize: '0.8em', color: '#888', marginTop: '4px' }}>
                        重試中... ({retryCount}/{maxRetries})
                    </div>
                )}
            </div>
        );
    }

    if (error) {
        return (
            <div className="error-pane" style={{ 
                padding: '15px', 
                backgroundColor: '#fff3f3', 
                border: '1px solid #ffc6c6',
                borderRadius: '6px',
                margin: '10px 0'
            }}>
                <div className="error-header" style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    marginBottom: '8px' 
                }}>
                    <span style={{ color: '#d63031', marginRight: '8px' }}>⚠️</span>
                    <strong style={{ color: '#d63031' }}>
                        {type === 'ngram' ? 'N-gram' : '詞頻'}分析載入失敗
                    </strong>
                </div>
                
                <div className="error-details" style={{ marginBottom: '12px' }}>
                    <p style={{ margin: '4px 0', color: '#666' }}>{error.message}</p>
                    {error.status && (
                        <p style={{ margin: '4px 0', fontSize: '0.85em', color: '#888' }}>
                            錯誤碼: {error.status}
                        </p>
                    )}
                    {error.retryCount > 0 && (
                        <p style={{ margin: '4px 0', fontSize: '0.85em', color: '#888' }}>
                            已重試: {error.retryCount} 次
                        </p>
                    )}
                </div>

                <div className="error-actions">
                    {error.canRetry && (
                        <button 
                            onClick={handleRetry}
                            style={{
                                padding: '6px 12px',
                                backgroundColor: '#0984e3',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                marginRight: '8px',
                                fontSize: '0.9em'
                            }}
                        >
                            重試載入
                        </button>
                    )}
                    {error.isNetworkError && (
                        <button 
                            onClick={() => window.location.reload()}
                            style={{
                                padding: '6px 12px',
                                backgroundColor: '#636e72',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '0.9em'
                            }}
                        >
                            重新載入頁面
                        </button>
                    )}
                </div>
            </div>
        );
    }

    // 降級策略
    if (!chartData.series || chartData.series.length === 0) {
        return (
            <div className="empty-state" style={{
                padding: '30px',
                textAlign: 'center',
                color: '#74b9ff',
                backgroundColor: '#f8f9fa'
            }}>
                <div style={{ fontSize: '2em', marginBottom: '10px' }}>
                    {type === 'ngram' ? '📝' : '🔤'}
                </div>
                <p>請選擇時間範圍以查看{type === 'ngram' ? 'N-gram' : '詞頻'}分析數據</p>
            </div>
        );
    }

    const chartEvents = {
        'click': onChartClick
    };

    return <ReactECharts 
        option={getOption()} 
        style={{ height: '300px', width: '100%', cursor: 'pointer' }} 
        onEvents={chartEvents} 
    />;
};

export default FrequencyChart; 