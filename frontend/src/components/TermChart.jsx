import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const TermChart = ({ data, range }) => {
    const chartData = React.useMemo(() => {
        if (!data) return null;

        const fromDate = range && range.from ? new Date(range.from) : null;
        const toDate = range && range.to ? new Date(range.to) : null;

        const termTotals = {};
        const dailyData = {};

        Object.keys(data).forEach(dateStr => {
            const currentDate = new Date(dateStr);
            if (!fromDate || (currentDate >= fromDate && currentDate <= toDate)) {
                const dayData = data[dateStr];
                dailyData[dateStr] = {};

                Object.keys(dayData).forEach(term => {
                    if (!term.includes(' ')) {
                        if (!termTotals[term]) termTotals[term] = 0;
                        termTotals[term] += dayData[term];
                        dailyData[dateStr][term] = dayData[term];
                    }
                });
            }
        });

        const top5Terms = Object.entries(termTotals)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(entry => entry[0]);

        if (top5Terms.length === 0) return [];

        const formattedData = Object.keys(dailyData).map(date => {
            const entry = { date };
            top5Terms.forEach(term => {
                entry[term] = dailyData[date][term] || 0;
            });
            return entry;
        }).sort((a, b) => new Date(a.date) - new Date(b.date));

        return { formattedData, top5Terms };

    }, [data, range]);

    if (!data) return <div className="loading-placeholder">正在等待詞頻資料...</div>;

    if (!chartData || chartData.top5Terms.length === 0) {
        return <div className="loading-placeholder">該時間範圍內無資料或無單詞資料</div>;
    }

    const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088FE"];

    return (
        <ResponsiveContainer width="100%" height={400}>
            <LineChart
                data={chartData.formattedData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                <XAxis dataKey="date" stroke="#8884d8" />
                <YAxis stroke="#8884d8" />
                <Tooltip
                    contentStyle={{
                        backgroundColor: 'rgba(30, 30, 50, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}
                />
                <Legend />
                {chartData.top5Terms.map((term, index) => (
                    <Line key={term} type="monotone" dataKey={term} stroke={colors[index % colors.length]} dot={false} />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );
};

export default TermChart; 