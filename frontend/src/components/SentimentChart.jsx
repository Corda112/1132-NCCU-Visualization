import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const SentimentChart = ({ data, range }) => {
    const filteredData = React.useMemo(() => {
        if (!data) return [];
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

    const chartData = React.useMemo(() => {
        if (filteredData.length === 0) return [];
        const sentimentByDay = filteredData.reduce((acc, curr) => {
            const day = curr.createdAt.split('T')[0];
            if (!acc[day]) {
                acc[day] = { totalSentiment: 0, count: 0 };
            }
            acc[day].totalSentiment += curr.sentiment;
            acc[day].count += 1;
            return acc;
        }, {});

        return Object.keys(sentimentByDay).map(day => ({
            date: day,
            sentiment: sentimentByDay[day].totalSentiment / sentimentByDay[day].count,
        })).sort((a, b) => new Date(a.date) - new Date(b.date));
    }, [filteredData]);

    if (!data) return <div className="loading-placeholder">正在等待情緒資料...</div>;

    if (chartData.length === 0) {
        return <div className="loading-placeholder">該時間範圍內無資料</div>;
    }

    return (
        <ResponsiveContainer width="100%" height={400}>
            <LineChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                <XAxis dataKey="date" stroke="#8884d8" />
                <YAxis stroke="#8884d8" domain={[-1, 1]} />
                <Tooltip
                    contentStyle={{
                        backgroundColor: 'rgba(30, 30, 50, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}
                />
                <Legend />
                <Line type="monotone" dataKey="sentiment" name="平均情緒" stroke="#82ca9d" dot={false} />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default SentimentChart; 