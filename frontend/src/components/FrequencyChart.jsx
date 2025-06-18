import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';
import { getApiUrl, API_ENDPOINTS, handleApiError } from '../config/api';

const FrequencyChart = ({ range, type, onTermSelect }) => {
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
                const response = await axios.get(getApiUrl(API_ENDPOINTS.TERM_NGRAM), {
                    params: { startDate, endDate },
                    timeout: 10000
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

            } catch (error) {
                const errorInfo = handleApiError(error);
                setError(errorInfo);
                console.error(`Error fetching ${type} data:`, error);
            }
            setLoading(false);
        };

        fetchData();
    }, [range, type]);

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
            onTermSelect(term, date); // params.name is the date on xAxis
        }
    };

    if (loading) {
        return <div className="loading-pane">載入{type === 'ngram' ? 'N-gram' : '詞頻'}資料中...</div>;
    }

    if (error) {
        return (
            <div className="error-pane">
                <p>載入{type === 'ngram' ? 'N-gram' : '詞頻'}資料失敗: {error.message}</p>
            </div>
        );
    }

    return <ReactECharts option={getOption()} style={{ height: '300px', width: '100%' }} onEvents={{ 'click': onChartClick }} />;
};

export default FrequencyChart; 