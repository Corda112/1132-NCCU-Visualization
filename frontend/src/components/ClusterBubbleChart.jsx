import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';
import { sentimentColors } from '../colors';

function ClusterBubbleChart({ range, selectedCluster, onSelect }) {
    const svgRef = useRef();
    const [nodes, setNodes] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!range || !range.from || !range.to) return;
        const startDate = new Date(range.from).toISOString().split('T')[0];
        const endDate = new Date(range.to).toISOString().split('T')[0];
        setLoading(true);
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
            .catch(err => console.error('Cluster fetch error', err))
            .finally(() => setLoading(false));
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
            .data(nodes, d => d.id)
            .enter()
            .append('circle')
            .attr('r', d => Math.sqrt(d.count) * 2 + 20)
            .attr('fill', d => sentimentColors[d.sentiment] || '#999')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5)
            .style('cursor', 'pointer')
            .on('click', (event, d) => onSelect && onSelect(d.id));

        const label = svg.selectAll('text')
            .data(nodes, d => d.id)
            .enter()
            .append('text')
            .text(d => d.id)
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .style('pointer-events', 'none')
            .style('fill', '#fff');

        simulation.on('tick', () => {
            node.attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('stroke', d => d.id === selectedCluster ? '#ff0' : '#fff')
                .attr('stroke-width', d => d.id === selectedCluster ? 3 : 1.5);
            label.attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        return () => simulation.stop();
    }, [nodes, selectedCluster, onSelect]);

    return (
        <div style={{ position: 'relative', width: '100%', height: '300px' }}>
            {loading && <div style={{position:'absolute',top:'50%',left:'50%',transform:'translate(-50%,-50%)',color:'#fff'}}>Loading...</div>}
            {!loading && nodes.length === 0 && <div style={{position:'absolute',top:'50%',left:'50%',transform:'translate(-50%,-50%)',color:'#fff'}}>No data</div>}
            <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
        </div>
    );
}

export default ClusterBubbleChart;
