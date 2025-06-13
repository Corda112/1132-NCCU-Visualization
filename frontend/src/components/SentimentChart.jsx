import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';

const SentimentChart = ({ range, onTermSelect }) => {
    const [chartData, setChartData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            if (!range || !range.from || !range.to) return;
            try {
                const startDate = new Date(range.from).toISOString().split('T')[0];
                const endDate = new Date(range.to).toISOString().split('T')[0];
                const response = await axios.get('http://localhost:3001/api/semantic', {
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
            } catch (error) {
                console.error('Error fetching sentiment data:', error);
            }
        };

        fetchData();
    }, [range]);

    const getOption = () => ({
        tooltip: {
            trigger: 'axis'
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
                data: chartData.map(item => item.Positive)
            },
            {
                name: 'Negative',
                type: 'line',
                data: chartData.map(item => item.Negative)
            },
            {
                name: 'Neutral',
                type: 'line',
                data: chartData.map(item => item.Neutral)
            }
        ]
    });

    const onChartClick = (params) => {
        if (onTermSelect) {
            onTermSelect({ sentiment: params.seriesName, date: params.name });
        }
    };

    return <ReactECharts option={getOption()} style={{ height: '300px', width: '100%' }} onEvents={{ 'click': onChartClick }} />;
};

export default SentimentChart; 