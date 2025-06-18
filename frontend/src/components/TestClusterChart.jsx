import React from 'react';
import ReactECharts from 'echarts-for-react';

const TestClusterChart = () => {
    console.log('TestClusterChart rendering...');
    
    const option = {
        backgroundColor: '#f0f0f0',
        title: {
            text: '測試聚類圖表',
            left: 'center',
            textStyle: {
                color: '#333'
            }
        },
        xAxis: {
            type: 'value',
            name: 'X軸',
            nameTextStyle: { color: '#333' },
            axisLabel: { color: '#333' }
        },
        yAxis: {
            type: 'value',
            name: 'Y軸',
            nameTextStyle: { color: '#333' },
            axisLabel: { color: '#333' }
        },
        series: [
            {
                name: '測試聚類 0',
                type: 'scatter',
                data: [
                    [10, 20, '測試文本1', 0],
                    [30, 40, '測試文本2', 0],
                    [50, 60, '測試文本3', 0]
                ],
                symbolSize: 10,
                itemStyle: {
                    color: '#ff4d4f'
                }
            },
            {
                name: '測試聚類 1',
                type: 'scatter',
                data: [
                    [70, 80, '測試文本4', 1],
                    [90, 100, '測試文本5', 1],
                    [110, 120, '測試文本6', 1]
                ],
                symbolSize: 10,
                itemStyle: {
                    color: '#52c41a'
                }
            }
        ],
        legend: {
            data: ['測試聚類 0', '測試聚類 1'],
            textStyle: {
                color: '#333'
            }
        },
        tooltip: {
            trigger: 'item',
            formatter: function(params) {
                return `聚類 ${params.data[3]}: ${params.data[2]}`;
            }
        }
    };

    console.log('TestClusterChart option:', option);

    return (
        <div style={{ padding: '20px' }}>
            <h3 style={{ color: '#333', marginBottom: '20px' }}>聚類圖表測試</h3>
            <div style={{ border: '2px solid blue', borderRadius: '8px', padding: '10px' }}>
                <ReactECharts 
                    option={option}
                    style={{ height: '400px', width: '100%' }}
                    onChartReady={(chart) => {
                        console.log('TestClusterChart ready:', chart);
                        console.log('Chart size:', chart.getWidth(), 'x', chart.getHeight());
                    }}
                />
            </div>
        </div>
    );
};

export default TestClusterChart; 