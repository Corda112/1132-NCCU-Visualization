import React, { useState, useEffect, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { apiClient, API_ENDPOINTS, handleApiError } from '../config/api';

const SentimentChart = ({ range, onTermSelect }) => {
    const [chartData, setChartData] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const [isProcessing, setIsProcessing] = useState(false); // 點擊處理狀態
    const [clickFeedback, setClickFeedback] = useState(null); // 點擊反饋狀態
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

    // 顯示點擊反饋 - 縮短顯示時間
    const showClickFeedback = (message, type = 'info') => {
        setClickFeedback({ message, type });
        setTimeout(() => {
            setClickFeedback(null);
        }, 1500);  // 從 2500ms 減少到 1500ms
    };

    const getOption = () => ({
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'line',
                lineStyle: {
                    color: '#1890ff',
                    width: 1,
                    type: 'dashed'
                },
                animation: false
            },
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            borderColor: '#d9d9d9',
            borderWidth: 1,
            textStyle: {
                color: '#333',
                fontSize: 12
            },
            formatter: function(params) {
                // 過濾掉透明系列的tooltip
                const validParams = params.filter(p => p.seriesName !== 'ClickableArea');
                const date = validParams[0]?.name;
                const total = validParams.reduce((sum, p) => sum + p.value, 0);
                
                return `
                    <div style="padding: 4px 0;">
                        <div style="font-weight: bold; margin-bottom: 6px;">${date}</div>
                        ${validParams.map(p => {
                            const percentage = total > 0 ? ((p.value / total) * 100).toFixed(1) : 0;
                            return `
                                <div style="margin: 3px 0; display: flex; justify-content: space-between; align-items: center;">
                                    <span>${p.marker}<span style="margin-left: 4px;">${p.seriesName}</span></span>
                                    <span style="margin-left: 12px; font-weight: bold;">${p.value} (${percentage}%)</span>
                                </div>
                            `;
                        }).join('')}
                        <div style="margin-top: 6px; padding-top: 4px; border-top: 1px solid #eee; font-size: 11px; color: #666;">
                            總計: ${total} 條推文
                        </div>
                        <div style="margin-top: 4px; font-size: 10px; color: #999;">
                            💡 點擊查看詳細內容
                        </div>
                    </div>
                `;
            }
        },
        legend: {
            data: ['Positive', 'Negative', 'Neutral'],
            top: 15,
            itemGap: 20,
            textStyle: {
                fontSize: 12,
                color: '#333'
            },
            itemStyle: {
                borderWidth: 0
            }
        },
        xAxis: {
            type: 'category',
            data: chartData.map(item => item.date),
            triggerEvent: true,
            axisLine: {
                lineStyle: {
                    color: '#d9d9d9'
                }
            },
            axisTick: {
                alignWithLabel: true,
                lineStyle: {
                    color: '#d9d9d9'
                }
            },
            axisLabel: {
                color: '#666',
                fontSize: 11
            }
        },
        yAxis: {
            type: 'value',
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            axisLabel: {
                color: '#666',
                fontSize: 11
            },
            splitLine: {
                lineStyle: {
                    color: '#f0f0f0'
                }
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '8%',
            top: '15%',
            containLabel: true
        },
        animation: false,
        series: [
            {
                name: 'Positive',
                type: 'line',
                data: chartData.map(item => item.Positive),
                smooth: true,
                lineStyle: { 
                    width: 3,
                    color: '#52c41a'
                },
                itemStyle: {
                    color: '#52c41a',
                    borderWidth: 0
                },
                emphasis: {
                    lineStyle: {
                        width: 4
                    }
                },
                triggerEvent: true,
                symbolSize: 0,
                showSymbol: false,
                hoverAnimation: false,
                animation: false,
                z: 2  // 線條在中層
            },
            {
                name: 'Negative',
                type: 'line',
                data: chartData.map(item => item.Negative),
                smooth: true,
                lineStyle: { 
                    width: 3,
                    color: '#ff4d4f'
                },
                itemStyle: {
                    color: '#ff4d4f',
                    borderWidth: 0
                },
                emphasis: {
                    lineStyle: {
                        width: 4
                    }
                },
                triggerEvent: true,
                symbolSize: 0,
                showSymbol: false,
                hoverAnimation: false,
                animation: false,
                z: 2  // 線條在中層
            },
            {
                name: 'Neutral',
                type: 'line',
                data: chartData.map(item => item.Neutral),
                smooth: true,
                lineStyle: { 
                    width: 3,
                    color: '#1890ff'
                },
                itemStyle: {
                    color: '#1890ff',
                    borderWidth: 0
                },
                emphasis: {
                    lineStyle: {
                        width: 4
                    }
                },
                triggerEvent: true,
                symbolSize: 0,
                showSymbol: false,
                hoverAnimation: false,
                animation: false,
                z: 2  // 線條在中層
            },
            // 簡化透明可點擊區域
            {
                name: 'ClickableArea',
                type: 'bar',
                data: chartData.map((item) => {
                    const maxValue = Math.max(item.Positive, item.Negative, item.Neutral);
                    return maxValue + Math.max(5, maxValue * 0.1);
                }),
                barWidth: '80%',
                itemStyle: {
                    color: 'transparent',
                    borderColor: 'transparent'
                },
                emphasis: {
                    itemStyle: {
                        color: 'rgba(24, 144, 255, 0.03)',
                        borderColor: 'transparent'
                    }
                },
                z: 3,  // 確保在最上層捕捉點擊
                silent: false,
                triggerEvent: true,
                tooltip: { show: false },
                legend: { show: false },
                animation: false
            }
        ]
    });

    const onChartClick = (params) => {
        console.log('🖱️ SentimentChart: Chart clicked!', params);
        console.log('🔍 Click details:', {
            componentType: params.componentType,
            seriesName: params.seriesName,
            name: params.name,
            value: params.value,
            dataIndex: params.dataIndex
        });
        
        if (!onTermSelect) {
            console.log('❌ onTermSelect not provided');
            return;
        }
        
        if (isProcessing) {
            console.log('⏳ Already processing, ignoring click');
            return;
        }
        
        // 設置處理狀態
        setIsProcessing(true);
        console.log('✅ Processing started');
        
        let dateStr = null;
        let sentiment = null;
        
        if (params.componentType === 'series') {
            dateStr = params.name;
            console.log('📊 Series clicked:', params.seriesName, 'for date:', dateStr);
            
            if (params.seriesName === 'ClickableArea') {
                sentiment = null;
                console.log('🎯 ClickableArea hit - loading all data for date');
                showClickFeedback(`載入 ${dateStr} 全部數據`, 'loading');
            } else if (['Positive', 'Negative', 'Neutral'].includes(params.seriesName)) {
                sentiment = params.seriesName;
                const sentimentText = sentiment === 'Positive' ? '正面' : sentiment === 'Negative' ? '負面' : '中性';
                console.log('📈 Sentiment line clicked:', sentiment);
                showClickFeedback(`載入${sentimentText}數據`, 'loading');
            }
        } else if (params.componentType === 'xAxis') {
            dateStr = params.value;
            sentiment = null;
            console.log('📅 X-axis clicked for date:', dateStr);
            showClickFeedback(`載入 ${dateStr} 全部數據`, 'loading');
        }
        
        if (dateStr) {
            const filterData = sentiment ? 
                { sentiment: sentiment, date: dateStr } : 
                { date: dateStr };
            
            console.log('📤 Sending filter data:', filterData);
            
            // 立即執行，無延遲
            onTermSelect(filterData);
            
            // 簡化的成功反饋
            setTimeout(() => {
                showClickFeedback('✓ 已載入', 'success');
                console.log('✅ Success feedback shown');
            }, 100);
        } else {
            console.log('⚠️ No valid date found in click event');
            showClickFeedback('請點擊線條或日期', 'warning');
        }
        
        // 快速重置處理狀態
        setTimeout(() => {
            setIsProcessing(false);
            console.log('🔄 Processing reset');
        }, 300);
    };

    if (loading) {
        return (
            <div style={{
                height: '350px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: '#fafafa',
                borderRadius: '8px',
                border: '1px solid #f0f0f0'
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
                <div style={{ color: '#666', fontSize: '14px' }}>載入情緒資料中...</div>
                {retryCount > 0 && (
                    <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                        重試中... ({retryCount}/{maxRetries})
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
                    <strong style={{ color: '#ff4d4f', fontSize: '16px' }}>情緒分析載入失敗</strong>
                </div>
                
                <div style={{ marginBottom: '16px' }}>
                    <p style={{ margin: '6px 0', color: '#666', fontSize: '14px' }}>{error.message}</p>
                    {error.status && (
                        <p style={{ margin: '4px 0', fontSize: '12px', color: '#999' }}>
                            錯誤碼: {error.status}
                        </p>
                    )}
                    {error.retryCount > 0 && (
                        <p style={{ margin: '4px 0', fontSize: '12px', color: '#999' }}>
                            已重試: {error.retryCount} 次
                        </p>
                    )}
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                    {error.canRetry && (
                        <button 
                            onClick={handleRetry}
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
                            重試載入
                        </button>
                    )}
                    {error.isNetworkError && (
                        <button 
                            onClick={() => window.location.reload()}
                            style={{
                                padding: '8px 16px',
                                backgroundColor: '#8c8c8c',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '14px',
                                transition: 'background-color 0.3s'
                            }}
                            onMouseOver={(e) => e.target.style.backgroundColor = '#a6a6a6'}
                            onMouseOut={(e) => e.target.style.backgroundColor = '#8c8c8c'}
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
            <div style={{
                height: '350px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: '#f6ffed',
                borderRadius: '8px',
                border: '1px solid #d9f7be',
                color: '#52c41a'
            }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>📊</div>
                <p style={{ fontSize: '16px', fontWeight: '500' }}>請選擇時間範圍以查看情緒分析數據</p>
                <p style={{ fontSize: '14px', color: '#8c8c8c', marginTop: '8px' }}>
                    選擇日期範圍後，點擊圖表線條查看詳細推文
                </p>
            </div>
        );
    }

    const chartEvents = {
        'click': onChartClick
    };

    return (
        <div style={{ position: 'relative' }}>
            {/* 點擊反饋提示 */}
            {clickFeedback && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    zIndex: 1000,
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '500',
                    maxWidth: '250px',
                    backgroundColor: clickFeedback.type === 'success' ? '#f6ffed' :
                                   clickFeedback.type === 'error' ? '#fff2f0' :
                                   clickFeedback.type === 'warning' ? '#fffbe6' : '#e6f7ff',
                    color: clickFeedback.type === 'success' ? '#52c41a' :
                           clickFeedback.type === 'error' ? '#ff4d4f' :
                           clickFeedback.type === 'warning' ? '#faad14' : '#1890ff',
                    border: `1px solid ${clickFeedback.type === 'success' ? '#d9f7be' :
                                        clickFeedback.type === 'error' ? '#ffccc7' :
                                        clickFeedback.type === 'warning' ? '#ffe58f' : '#bae7ff'}`,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                    animation: 'slideInRight 0.3s ease-out'
                }}>
                    {clickFeedback.type === 'loading' && (
                        <span style={{ marginRight: '6px' }}>⏳</span>
                    )}
                    {clickFeedback.type === 'success' && (
                        <span style={{ marginRight: '6px' }}>✅</span>
                    )}
                    {clickFeedback.type === 'error' && (
                        <span style={{ marginRight: '6px' }}>❌</span>
                    )}
                    {clickFeedback.type === 'warning' && (
                        <span style={{ marginRight: '6px' }}>⚠️</span>
                    )}
                    {clickFeedback.message}
                </div>
            )}

            {/* 處理狀態覆蓋層 */}
            {isProcessing && (
                <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(255, 255, 255, 0.7)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    zIndex: 999,
                    borderRadius: '8px'
                }}>
                    <div style={{
                        padding: '12px 20px',
                        backgroundColor: 'white',
                        borderRadius: '6px',
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                        border: '1px solid #d9d9d9',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}>
                        <div style={{
                            width: '16px',
                            height: '16px',
                            border: '2px solid #f3f3f3',
                            borderTop: '2px solid #1890ff',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite'
                        }}></div>
                        <span style={{ color: '#666', fontSize: '14px' }}>處理中...</span>
                    </div>
                </div>
            )}

            <ReactECharts 
                option={getOption()} 
                style={{ 
                    height: '350px', 
                    width: '100%', 
                    cursor: isProcessing ? 'wait' : 'pointer',
                    borderRadius: '8px',
                    border: '1px solid #f0f0f0'
                }} 
                onEvents={chartEvents} 
            />

            <style jsx>{`
                @keyframes slideInRight {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
};

export default SentimentChart; 