import React, { useState, useEffect, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { apiClient, API_ENDPOINTS, handleApiError } from '../config/api';

const FrequencyChart = ({ range, type, onTermSelect }) => {
    const [chartData, setChartData] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const [isProcessing, setIsProcessing] = useState(false);
    const [clickFeedback, setClickFeedback] = useState({ show: false, message: '', type: 'info' });
    const maxRetries = 3;

    // 統一的反饋機制
    const showClickFeedback = (message, type = 'info') => {
        setClickFeedback({ show: true, message, type });
        setTimeout(() => {
            setClickFeedback({ show: false, message: '', type: 'info' });
        }, 1500);
    };

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

            const sortedDates = Object.keys(dailyData).sort();
            const processedData = sortedDates.map(date => ({
                date,
                ...dailyData[date]
            }));

            setChartData({
                dates: sortedDates,
                series: top10.map(term => ({
                    name: term,
                    type: 'line',
                    data: processedData.map(d => d[term] || 0)
                })),
                top10terms: top10,
                processedData
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

    // 統一的圖表配置
    const getOption = () => {
        if (!chartData.series || chartData.series.length === 0) {
            return {};
        }

        // 計算最大值用於 ClickableArea
        const maxValues = chartData.processedData?.map(item => {
            const values = chartData.top10terms.map(term => item[term] || 0);
            return Math.max(...values);
        }) || [];

        return {
            tooltip: {
                trigger: 'axis',
                formatter: function(params) {
                    const date = params[0].axisValue;
                    const termName = type === 'ngram' ? 'N-gram' : '術語';
                    let content = `<div style="margin-bottom: 8px; font-weight: 600;">${date}</div>`;
                    
                    params.forEach(param => {
                        if (param.seriesName !== 'ClickableArea') {
                            content += `<div style="margin: 4px 0;">
                                ${param.marker}
                                <span style="font-weight: 500;">${param.seriesName}</span>: 
                                <span style="color: #1890ff; font-weight: 600;">${param.value}</span> 次
                            </div>`;
                        }
                    });
                    
                    content += `<div style="margin-top: 8px; font-size: 12px; color: #8c8c8c;">
                        點擊查看「${termName}」相關文章
                    </div>`;
                    
                    return content;
                }
            },
            legend: {
                data: chartData.top10terms,
                orient: 'vertical',
                align: 'left',
                left: 'right',
                type: 'scroll',
                textStyle: {
                    fontSize: 12
                }
            },
            xAxis: {
                type: 'category',
                data: chartData.dates,
                triggerEvent: true,
                axisLabel: {
                    fontSize: 11,
                    color: '#666'
                }
            },
            yAxis: {
                type: 'value',
                name: '頻次',
                nameTextStyle: {
                    fontSize: 12,
                    color: '#666'
                },
                axisLabel: {
                    fontSize: 11,
                    color: '#666'
                }
            },
            grid: {
                left: '10%',
                right: '25%',
                bottom: '15%',
                top: '15%'
            },
            series: [
                // 原始數據線條
                ...chartData.series.map((s, index) => ({
                    ...s,
                    smooth: true,
                    lineStyle: { 
                        width: 3,
                        color: undefined // 使用默認調色盤
                    },
                    itemStyle: {
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
                })),
                // 透明可點擊區域
                {
                    name: 'ClickableArea',
                    type: 'bar',
                    data: maxValues.map((maxValue) => {
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
        };
    };

    // 統一的點擊處理邏輯
    const onChartClick = (params) => {
        console.log('🖱️ FrequencyChart: Chart clicked!', params);
        console.log('🔍 Click details:', {
            componentType: params.componentType,
            seriesName: params.seriesName,
            name: params.name,
            value: params.value,
            dataIndex: params.dataIndex,
            type: type
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
        let term = null;
        
        if (params.componentType === 'series') {
            dateStr = params.name;
            console.log('📊 Series clicked:', params.seriesName, 'for date:', dateStr);
            
            if (params.seriesName === 'ClickableArea') {
                term = null;
                console.log('🎯 ClickableArea hit - loading all data for date');
                showClickFeedback(`載入 ${dateStr} 全部${type === 'ngram' ? 'N-gram' : '術語'}`, 'loading');
            } else if (chartData.top10terms.includes(params.seriesName)) {
                term = params.seriesName;
                console.log('📈 Term line clicked:', term);
                showClickFeedback(`載入「${term}」數據`, 'loading');
            }
        } else if (params.componentType === 'xAxis') {
            dateStr = params.value;
            term = null;
            console.log('📅 X-axis clicked for date:', dateStr);
            showClickFeedback(`載入 ${dateStr} 全部${type === 'ngram' ? 'N-gram' : '術語'}`, 'loading');
        }
        
        if (dateStr) {
            const filterData = term ? 
                { term: term, date: dateStr } : 
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
            showClickFeedback(`請點擊${type === 'ngram' ? 'N-gram' : '術語'}線條或日期`, 'warning');
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
                <div style={{ color: '#666', fontSize: '14px' }}>
                    載入{type === 'ngram' ? 'N-gram' : '術語'}資料中...
                </div>
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
                    <strong style={{ color: '#ff4d4f', fontSize: '16px' }}>
                        {type === 'ngram' ? 'N-gram' : '術語'}分析載入失敗
                    </strong>
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
    if (!chartData.series || chartData.series.length === 0) {
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
                <p style={{ fontSize: '16px', fontWeight: '500', marginBottom: '8px' }}>
                    暫無{type === 'ngram' ? 'N-gram' : '術語'}數據
                </p>
                <p style={{ fontSize: '14px', color: '#8c8c8c' }}>
                    請選擇時間範圍以查看{type === 'ngram' ? 'N-gram' : '術語'}分析數據
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
            {clickFeedback.show && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    backgroundColor: clickFeedback.type === 'success' ? '#f6ffed' : 
                                   clickFeedback.type === 'loading' ? '#e6f7ff' : '#fff7e6',
                    color: clickFeedback.type === 'success' ? '#52c41a' : 
                           clickFeedback.type === 'loading' ? '#1890ff' : '#d46b08',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    border: `1px solid ${clickFeedback.type === 'success' ? '#d9f7be' : 
                                        clickFeedback.type === 'loading' ? '#bae7ff' : '#ffd591'}`,
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                    zIndex: 1000,
                    animation: 'fadeInOut 1.5s ease-in-out',
                    pointerEvents: 'none'
                }}>
                    {clickFeedback.message}
                </div>
            )}
            
            <ReactECharts 
                option={getOption()} 
                style={{ height: '350px', width: '100%', cursor: 'pointer' }} 
                onEvents={chartEvents} 
            />
            
            <style jsx>{`
                @keyframes fadeInOut {
                    0% { opacity: 0; transform: translateY(-10px); }
                    20% { opacity: 1; transform: translateY(0); }
                    80% { opacity: 1; transform: translateY(0); }
                    100% { opacity: 0; transform: translateY(-10px); }
                }
            `}</style>
        </div>
    );
};

export default FrequencyChart; 