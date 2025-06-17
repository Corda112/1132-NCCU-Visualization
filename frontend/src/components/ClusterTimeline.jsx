import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

function ClusterTimeline({ range, onBrush }) {
    const svgRef = useRef();
    const [stacked, setStacked] = useState([]);
    const [keys, setKeys] = useState([]);

    useEffect(() => {
        if (!range || !range.from || !range.to) return;
        const startDate = new Date(range.from).toISOString().split('T')[0];
        const endDate = new Date(range.to).toISOString().split('T')[0];
        axios.get('http://localhost:3001/api/clusters', { params: { startDate, endDate } })
            .then(res => {
                const formatMonth = d3.timeFormat('%Y-%m');
                const grouped = d3.rollups(res.data,
                    v => v.length,
                    d => formatMonth(new Date(d.createdAt)),
                    d => d.cluster_id);
                const months = Array.from(new Set(res.data.map(d => formatMonth(new Date(d.createdAt))))).sort();
                const clusters = Array.from(new Set(res.data.map(d => d.cluster_id))).sort((a,b)=>a-b);
                const seriesData = months.map(month => {
                    const entry = { month };
                    clusters.forEach(c => entry[c] = 0);
                    return entry;
                });
                grouped.forEach(([month, arr]) => {
                    const obj = seriesData.find(d => d.month === month);
                    arr.forEach(([cluster, count]) => { obj[cluster] = count; });
                });
                setStacked(seriesData);
                setKeys(clusters);
            })
            .catch(err => console.error('Timeline fetch error', err));
    }, [range]);

    useEffect(() => {
        if (!svgRef.current || stacked.length === 0) return;
        const width = svgRef.current.clientWidth || 400;
        const height = 300;
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const x = d3.scaleBand().domain(stacked.map(d => d.month)).range([40, width - 20]).padding(0.1);
        const y = d3.scaleLinear().range([height - 30, 20]);
        const stack = d3.stack().keys(keys);
        const series = stack(stacked);
        const maxY = d3.max(series, s => d3.max(s, d => d[1])) || 0;
        y.domain([0, maxY]).nice();
        const color = d3.scaleOrdinal(d3.schemeCategory10).domain(keys);

        svg.append('g').attr('transform', `translate(0,${height - 30})`).call(d3.axisBottom(x).tickSizeOuter(0))
            .selectAll('text').attr('transform', 'rotate(-40)').attr('text-anchor', 'end');
        svg.append('g').attr('transform', 'translate(40,0)').call(d3.axisLeft(y));

        const area = d3.area()
            .x(d => x(d.data.month) + x.bandwidth()/2)
            .y0(d => y(d[0]))
            .y1(d => y(d[1]));

        svg.selectAll('path.layer')
            .data(series)
            .enter()
            .append('path')
            .attr('class','layer')
            .attr('fill', d => color(d.key))
            .attr('d', area);

        const brush = d3.brushX().extent([[40,20],[width-20,height-30]])
            .on('end', e => {
                if (!e.selection) return;
                const [x0,x1] = e.selection;
                const monthScale = d3.scaleQuantize().domain([40,width-20]).range(stacked.map(d=>d.month));
                const start = monthScale(x0);
                const end = monthScale(x1);
                if (onBrush) onBrush({start,end});
            });
        svg.append('g').call(brush);
    }, [stacked, keys, onBrush]);

    return <svg ref={svgRef} style={{ width: '100%', height: '300px' }} />;
}

export default ClusterTimeline;
