import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const NgramChart = ({ data, range }) => {
    const chartData = React.useMemo(() => {
        if (!data) return null;

        const fromDate = range && range.from ? new Date(range.from) : null;
        const toDate = range && range.to ? new Date(range.to) : null;

        const ngramTotals = {};
        const dailyData = {};

        Object.keys(data).forEach(dateStr => {
            const currentDate = new Date(dateStr);
            if (!fromDate || (currentDate >= fromDate && currentDate <= toDate)) {
                const dayData = data[dateStr];
                dailyData[dateStr] = {};

                // Process n-grams (assuming they contain spaces)
                Object.keys(dayData).forEach(term => {
                    if (term.includes(' ')) {
                        if (!ngramTotals[term]) ngramTotals[term] = 0;
                        ngramTotals[term] += dayData[term];
                        dailyData[dateStr][term] = dayData[term];
                    }
                });
            }
        });

        const top5Ngrams = Object.entries(ngramTotals)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(entry => entry[0]);

        if (top5Ngrams.length === 0) return [];

        const formattedData = Object.keys(dailyData).map(date => {
            const entry = { date };
            top5Ngrams.forEach(term => {
                entry[term] = dailyData[date][term] || 0;
            });
            return entry;
        }).sort((a, b) => new Date(a.date) - new Date(b.date));

        return { formattedData, top5Ngrams };

    }, [data, range]);

    if (!data) return <div className="loading-placeholder">正在等待 N-gram 詞頻資料...</div>;

    if (!chartData || chartData.top5Ngrams.length === 0) {
        return <div className="loading-placeholder">該時間範圍內無 N-gram 資料</div>;
    }

    const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088FE"].reverse();

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
                {chartData.top5Ngrams.map((term, index) => (
                    <Line key={term} type="monotone" dataKey={term} stroke={colors[index % colors.length]} dot={false} />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );
};

export default NgramChart; 