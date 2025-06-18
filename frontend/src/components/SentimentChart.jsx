import React, { useState, useEffect, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { apiClient, API_ENDPOINTS, handleApiError } from '../config/api';

const SentimentChart = ({ range, onTermSelect }) => {
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
            const response = await apiClient.get(API_ENDPOINTS.SEMANTIC, {
                params: { startDate, endDate }
            });

            // Process data: count sentiment types per day
            const processedData = response.data.reduce((acc, { createdAt, sentiment }) => {
                const date = new Date(createdAt).toISOString().split('T')[0];
                if (!acc[date]) {
                    acc[date] = { Positive: 0, Negative: 0, Neutral: 0 };
                }
                if (sentiment === 'Positive' || sentiment === 'Negative' || sentiment === 'Neutral') {
                    acc[date][sentiment]++;
                }
                return acc;
            }, {});

            const chartSeries = Object.keys(processedData).map(date => ({
                date,
                ...processedData[date]
            }));

            setChartData(chartSeries);
            setError(null);
            setRetryCount(0);
        } catch (error) {
            console.error('Error fetching sentiment data:', error);
            const errorInfo = handleApiError(error);
            
            // 錯誤統計和重試邏輯
            if (errorInfo.isNetworkError && retryCount < maxRetries) {
                setRetryCount(prev => prev + 1);
                setTimeout(() => {
                    fetchData(true);
                }, 1000 * Math.pow(2, retryCount)); // 指數退避
            } else {
                setError({
                    ...errorInfo,
                    retryCount,
                    canRetry: retryCount < maxRetries,
                    component: 'SentimentChart'
                });
            }
        }
        setLoading(false);
    }, [range, retryCount]);

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
                return `${params[0].name}<br/>${params.map(p => 
                    `${p.marker}${p.seriesName}: ${p.value}`
                ).join('<br/>')}`;
            }
        },
        legend: {
            data: ['Positive', 'Negative', 'Neutral']
        },
        xAxis: {
            type: 'category',
            data: chartData.map(item => item.date)
        },
        yAxis: {
            type: 'value'
        },
        series: [
            {
                name: 'Positive',
                type: 'line',
                data: chartData.map(item => item.Positive),
                smooth: true,
                lineStyle: { width: 3 },
                emphasis: {
                    focus: 'series',
                    blurScope: 'coordinateSystem'
                }
            },
            {
                name: 'Negative',
                type: 'line',
                data: chartData.map(item => item.Negative),
                smooth: true,
                lineStyle: { width: 3 },
                emphasis: {
                    focus: 'series',
                    blurScope: 'coordinateSystem'
                }
            },
            {
                name: 'Neutral',
                type: 'line',
                data: chartData.map(item => item.Neutral),
                smooth: true,
                lineStyle: { width: 3 },
                emphasis: {
                    focus: 'series',
                    blurScope: 'coordinateSystem'
                }
            }
        ]
    });

    const onChartClick = (params) => {
        console.log('SentimentChart: Chart clicked!', params);
        if (onTermSelect) {
            const dateStr = params.name;
            const filterData = { 
                sentiment: params.seriesName, 
                date: dateStr 
            };
            console.log('SentimentChart: Sending filter data:', filterData);
            onTermSelect(filterData);
        }
    };

    if (loading) {
        return (
            <div className="loading-pane">
                <div>載入情緒資料中...</div>
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
                    <strong style={{ color: '#d63031' }}>情緒分析載入失敗</strong>
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

    // 降級策略：如果沒有數據但也沒有錯誤，顯示提示
    if (chartData.length === 0) {
        return (
            <div className="empty-state" style={{
                padding: '30px',
                textAlign: 'center',
                color: '#74b9ff',
                backgroundColor: '#f8f9fa'
            }}>
                <div style={{ fontSize: '2em', marginBottom: '10px' }}>📊</div>
                <p>請選擇時間範圍以查看情緒分析數據</p>
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

export default SentimentChart; 