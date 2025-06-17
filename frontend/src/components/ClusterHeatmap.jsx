import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

function ClusterHeatmap({ range, selectedCluster }) {
    const svgRef = useRef();
    const [matrix, setMatrix] = useState([]);
    const [clusters, setClusters] = useState([]);
    const [months, setMonths] = useState([]);

    useEffect(() => {
        if (!range || !range.from || !range.to) return;
        const startDate = new Date(range.from).toISOString().split('T')[0];
        const endDate = new Date(range.to).toISOString().split('T')[0];
        axios.get('http://localhost:3001/api/clusters', { params: { startDate, endDate } })
            .then(res => {
                const formatMonth = d3.timeFormat('%Y-%m');
                const monthsArr = Array.from(new Set(res.data.map(d => formatMonth(new Date(d.createdAt))))).sort();
                const clustersArr = Array.from(new Set(res.data.map(d => d.cluster_id))).sort((a,b)=>a-b);
                const matrixData = monthsArr.map(month => {
                    const row = { month };
                    clustersArr.forEach(c => { row[c] = 0; });
                    return row;
                });
                const grouped = d3.rollups(res.data, v => v.length, d => formatMonth(new Date(d.createdAt)), d => d.cluster_id);
                grouped.forEach(([m, arr]) => {
                    const row = matrixData.find(d => d.month === m);
                    arr.forEach(([c, count]) => { row[c] = count; });
                });
                setMatrix(matrixData);
                setClusters(clustersArr);
                setMonths(monthsArr);
            })
            .catch(err => console.error('Heatmap fetch error', err));
    }, [range]);

    useEffect(() => {
        if (!svgRef.current || matrix.length === 0) return;
        const width = svgRef.current.clientWidth || 400;
        const height = 300;
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const gridWidth = (width - 80) / months.length;
        const gridHeight = (height - 40) / clusters.length;
        const color = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, d3.max(matrix, row => d3.max(clusters, c => row[c])) || 1]);

        const g = svg.append('g').attr('transform','translate(60,20)');

    g.selectAll('g.row')
        .data(matrix)
        .enter()
        .append('g')
        .attr('class', 'row')
        .attr('transform', (d, i) => `translate(0,${i * gridHeight})`)
        .each(function (row) {
            const cellG = d3.select(this);
            cellG.selectAll('rect')
                .data(clusters.map(c => ({ cluster: c, value: row[c] })))
                .enter()
                .append('rect')
                .attr('x', (d, j) => j * gridWidth)
                .attr('width', gridWidth)
                .attr('height', gridHeight)
                .attr('fill', d => color(d.value))
                .attr('stroke', d => selectedCluster === d.cluster ? '#f00' : 'none')
                .attr('stroke-width', d => selectedCluster === d.cluster ? 2 : 0)
                .append('title')
                .text(d => d.value);
        });

        svg.append('g').attr('transform',`translate(60,${height-20})`)
            .call(d3.axisBottom(d3.scaleBand().domain(months).range([0, months.length*gridWidth])))
            .selectAll('text').attr('transform','rotate(-40)').attr('text-anchor','end');
        svg.append('g').attr('transform','translate(60,20)')
            .call(d3.axisLeft(d3.scaleBand().domain(clusters).range([0, clusters.length*gridHeight])));
    }, [matrix, clusters, months, selectedCluster]);

    return <svg ref={svgRef} style={{ width: '100%', height: '300px' }} />;
}

export default ClusterHeatmap;
