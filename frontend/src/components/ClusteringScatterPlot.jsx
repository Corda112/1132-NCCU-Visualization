import React, { useState, useEffect, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { apiClient, API_ENDPOINTS, handleApiError } from '../config/api';

const ClusteringScatterPlot = ({ range, onTermSelect }) => {
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
            const response = await apiClient.get(API_ENDPOINTS.CLUSTERS, {
                params: { startDate, endDate }
            });
            setChartData(response.data);
            setError(null);
            setRetryCount(0);
        } catch (error) {
            console.error('Error fetching clustering data:', error);
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
                    component: 'ClusteringScatterPlot'
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

    const getOption = () => {
        const clusters = [...new Set(chartData.map(item => item.cluster_id))];
        const series = clusters.map(clusterId => ({
            name: `Cluster ${clusterId}`,
            type: 'scatter',
            data: chartData
                .filter(item => item.cluster_id === clusterId)
                .map(item => [item.x, item.y, item.cleaned_text]),
            emphasis: {
                focus: 'series',
                label: {
                    show: true,
                    formatter: function (param) {
                        return param.data[2].substring(0, 50) + '...';
                    },
                    position: 'top'
                }
            },
        }));

        return {
            tooltip: {
                trigger: 'item',
                formatter: function (params) {
                    return `<b>${params.seriesName}</b><br/>Text: ${params.data[2].substring(0, 100)}...`;
                }
            },
            legend: {
                data: clusters.map(id => `Cluster ${id}`),
                orient: 'vertical',
                align: 'left',
                left: 'right',
            },
            xAxis: { type: 'value', name: 'X' },
            yAxis: { type: 'value', name: 'Y' },
            series: series
        };
    };

    const onChartClick = (params) => {
        console.log('ClusteringScatterPlot: Chart clicked!', params);
        if (onTermSelect && params.data && params.data[2]) {
            const text = params.data[2];
            const searchTerm = text.split(' ').slice(0, 3).join(' ');
            const filterData = { term: searchTerm };
            console.log('ClusteringScatterPlot: Sending filter data:', filterData);
            onTermSelect(filterData);
        }
    };

    if (loading) {
        return (
            <div className="loading-pane">
                <div>載入聚類資料中...</div>
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
                    <strong style={{ color: '#d63031' }}>聚類分析載入失敗</strong>
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
    if (chartData.length === 0) {
        return (
            <div className="empty-state" style={{
                padding: '30px',
                textAlign: 'center',
                color: '#74b9ff',
                backgroundColor: '#f8f9fa'
            }}>
                <div style={{ fontSize: '2em', marginBottom: '10px' }}>🎯</div>
                <p>請選擇時間範圍以查看聚類分析數據</p>
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

export default ClusteringScatterPlot; 