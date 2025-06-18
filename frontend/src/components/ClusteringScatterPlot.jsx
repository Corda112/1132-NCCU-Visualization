import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import axios from 'axios';
import { getApiUrl, API_ENDPOINTS, handleApiError } from '../config/api';

const ClusteringScatterPlot = ({ range, onTermSelect }) => {
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
                const response = await axios.get(getApiUrl(API_ENDPOINTS.CLUSTERS), {
                    params: { startDate, endDate },
                    timeout: 10000
                });
                setChartData(response.data);
            } catch (error) {
                const errorInfo = handleApiError(error);
                setError(errorInfo);
                console.error('Error fetching clustering data:', error);
            }
            setLoading(false);
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

    const onChartClick = (params) => {
        console.log('ClusteringScatterPlot: Chart clicked!', params);
        if (onTermSelect && params.data && params.data[2]) {
            // 使用文章內容的前幾個單詞作為搜索詞
            const text = params.data[2];
            const searchTerm = text.split(' ').slice(0, 3).join(' ');
            const filterData = { term: searchTerm };
            console.log('ClusteringScatterPlot: Sending filter data:', filterData);
            onTermSelect(filterData);
        }
    };

    if (loading) {
        return <div className="loading-pane">載入聚類資料中...</div>;
    }

    if (error) {
        return (
            <div className="error-pane">
                <p>載入聚類資料失敗: {error.message}</p>
            </div>
        );
    }

    return <ReactECharts option={getOption()} style={{ height: '300px', width: '100%' }} onEvents={{ 'click': onChartClick }} />;
};

export default ClusteringScatterPlot; 