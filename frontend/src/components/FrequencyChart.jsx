import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';

const FrequencyChart = ({ range, type, onTermSelect }) => {
    const [chartData, setChartData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            if (!range || !range.from || !range.to) return;
            try {
                const startDate = new Date(range.from).toISOString().split('T')[0];
                const endDate = new Date(range.to).toISOString().split('T')[0];
                const response = await axios.get('http://localhost:3001/api/term-ngram', {
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

            } catch (error) {
                console.error(`Error fetching ${type} data:`, error);
            }
        };

        fetchData();
    }, [range, type]);

    const getOption = () => ({
        tooltip: {
            trigger: 'axis'
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
        series: chartData.series
    });

    const onChartClick = (params) => {
        if (onTermSelect) {
            onTermSelect(params.seriesName, params.name); // params.name is the date on xAxis
        }
    };

    return <ReactECharts option={getOption()} style={{ height: '300px', width: '100%' }} onEvents={{ 'click': onChartClick }} />;
};

export default FrequencyChart; 