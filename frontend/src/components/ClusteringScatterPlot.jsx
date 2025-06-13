import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';

const ClusteringScatterPlot = ({ range }) => {
    const [chartData, setChartData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            if (!range || !range.from || !range.to) return;
            try {
                const startDate = new Date(range.from).toISOString().split('T')[0];
                const endDate = new Date(range.to).toISOString().split('T')[0];
                const response = await axios.get('http://localhost:3001/api/clusters', {
                    params: { startDate, endDate }
                });
                setChartData(response.data);
            } catch (error) {
                console.error('Error fetching clustering data:', error);
            }
        };

        fetchData();
    }, [range]);

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
                        return param.data[2].substring(0, 50) + '...'; // Show part of the text on hover
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

    return <ReactECharts option={getOption()} style={{ height: '300px', width: '100%' }} />;
};

export default ClusteringScatterPlot; 