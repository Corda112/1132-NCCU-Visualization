import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ClusterChart = ({ data, range }) => {
    const filteredData = React.useMemo(() => {
        if (!data) return []; // Handle null case inside the hook
        if (!range || !range.from || !range.to) {
            return data;
        }
        const fromDate = new Date(range.from);
        const toDate = new Date(range.to);
        return data.filter(item => {
            const itemDate = new Date(item.createdAt);
            return itemDate >= fromDate && itemDate <= toDate;
        });
    }, [data, range]);

    if (!data) return <div className="loading-placeholder">正在等待分群資料...</div>;

    const clusterColors = [
        "#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088FE",
        "#00C49F", "#FFBB28", "#FF8042", "#FF4444", "#4B0082"
    ];

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const point = payload[0].payload;
            return (
                <div className="custom-tooltip" style={{
                    backgroundColor: 'rgba(30, 30, 50, 0.9)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    padding: '10px',
                    borderRadius: '8px'
                }}>
                    <p style={{ color: '#fff', marginBottom: '5px' }}>{`"${point.cleaned_text.substring(0, 100)}..."`}</p>
                    <p style={{ color: '#8884d8' }}>{`Cluster ID: ${point.cluster_id}`}</p>
                    <p style={{ color: '#82ca9d' }}>{`Sentiment: ${point.sentiment.toFixed(3)}`}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <ResponsiveContainer width="100%" height={400}>
            <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                <XAxis type="number" dataKey="x" name="x" stroke="#8884d8" />
                <YAxis type="number" dataKey="y" name="y" stroke="#8884d8" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
                <Scatter name="Posts" data={filteredData} fill="#8884d8">
                    {
                        filteredData.map((entry, index) => (
                            <ZAxis key={`cell-${index}`} dataKey="cluster_id" fill={clusterColors[entry.cluster_id % clusterColors.length]} />
                        ))
                    }
                </Scatter>
            </ScatterChart>
        </ResponsiveContainer>
    );
};

export default ClusterChart; 