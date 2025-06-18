import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';
import { getApiUrl, API_ENDPOINTS, handleApiError } from '../config/api';

const SentimentChart = ({ range, onTermSelect }) => {
    const [chartData, setChartData] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            if (!range || !range.from || !range.to) return;
            
            setLoading(true);
            setError(null);
            
            try {
                const startDate = new Date(range.from).toISOString().split('T')[0];
                const endDate = new Date(range.to).toISOString().split('T')[0];
                const response = await axios.get(getApiUrl(API_ENDPOINTS.SEMANTIC), {
                    params: { startDate, endDate },
                    timeout: 10000
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
            } catch (error) {
                const errorInfo = handleApiError(error);
                setError(errorInfo);
                console.error('Error fetching sentiment data:', error);
            }
            setLoading(false);
        };

        fetchData();
    }, [range]);

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
                lineStyle: { width: 3 }
            },
            {
                name: 'Negative',
                type: 'line',
                data: chartData.map(item => item.Negative),
                smooth: true,
                lineStyle: { width: 3 }
            },
            {
                name: 'Neutral',
                type: 'line',
                data: chartData.map(item => item.Neutral),
                smooth: true,
                lineStyle: { width: 3 }
            }
        ]
    });

    const onChartClick = (params) => {
        console.log('SentimentChart: Chart clicked!', params);
        if (onTermSelect) {
            const filterData = { sentiment: params.seriesName, date: params.name };
            console.log('SentimentChart: Sending filter data:', filterData);
            onTermSelect(filterData);
        }
    };

    if (loading) {
        return <div className="loading-pane">載入情緒資料中...</div>;
    }

    if (error) {
        return (
            <div className="error-pane">
                <p>載入情緒資料失敗: {error.message}</p>
            </div>
        );
    }

    return <ReactECharts option={getOption()} style={{ height: '300px', width: '100%' }} onEvents={{ 'click': onChartClick }} />;
};

export default SentimentChart; 