import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

const sentimentColors = {
    Positive: '#2ca02c',
    Negative: '#d62728',
    Neutral: '#1f77b4'
};

function ClusterBubbleChart({ range, selectedCluster, onSelect }) {
    const svgRef = useRef();
    const [nodes, setNodes] = useState([]);

    useEffect(() => {
        if (!range || !range.from || !range.to) return;
        const startDate = new Date(range.from).toISOString().split('T')[0];
        const endDate = new Date(range.to).toISOString().split('T')[0];
        axios.get('http://localhost:3001/api/clusters', { params: { startDate, endDate } })
            .then(res => {
                const grouped = d3.group(res.data, d => d.cluster_id);
                const processed = Array.from(grouped, ([id, items]) => {
                    const sentimentCount = d3.rollup(items, v => v.length, d => d.sentiment);
                    let predominant = 'Neutral';
                    let max = 0;
                    for (const [s, c] of sentimentCount.entries()) {
                        if (c > max) { max = c; predominant = s; }
                    }
                    return { id, count: items.length, sentiment: predominant };
                });
                setNodes(processed);
            })
            .catch(err => console.error('Cluster fetch error', err));
    }, [range]);

    useEffect(() => {
        if (!svgRef.current) return;
        const width = svgRef.current.clientWidth || 400;
        const height = svgRef.current.clientHeight || 400;
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const simulation = d3.forceSimulation(nodes)
            .force('charge', d3.forceManyBody().strength(5))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => Math.sqrt(d.count) * 2 + 20));

        const node = svg.selectAll('circle')
            .data(nodes)
            .enter()
            .append('circle')
            .attr('r', d => Math.sqrt(d.count) * 2 + 20)
            .attr('fill', d => sentimentColors[d.sentiment] || '#999')
            .attr('stroke', d => d.id === selectedCluster ? '#ff0' : '#fff')
            .attr('stroke-width', d => d.id === selectedCluster ? 3 : 1.5)
            .on('click', (event, d) => onSelect && onSelect(d.id));

        const label = svg.selectAll('text')
            .data(nodes)
            .enter()
            .append('text')
            .text(d => d.id)
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .style('pointer-events', 'none')
            .style('fill', '#fff');

        simulation.on('tick', () => {
            node.attr('cx', d => d.x)
                .attr('cy', d => d.y);
            label.attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        return () => simulation.stop();
    }, [nodes, selectedCluster, onSelect]);

    return (
        <svg ref={svgRef} style={{ width: '100%', height: '300px' }} />
    );
}

export default ClusterBubbleChart;
