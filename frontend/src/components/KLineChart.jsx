import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

function KLineChart({ onRangeChange }) {
    const d3Container = useRef(null);
    const [data, setData] = useState([]);
    const [range, setRange] = useState([0, 0]);

    useEffect(() => {
        axios.get('http://localhost:3001/api/kline').then(res => {
            const d = res.data.map(row => ({
                ...row,
                date: new Date(row.timestamp || row.date || row.Date),
                open: +row.Open,
                high: +row.High,
                low: +row.Low,
                close: +row.Close,
                volume: +row.Volume
            })).sort((a, b) => a.date - b.date);
            setData(d);
            setRange([0, d.length - 1]);
        });
    }, []);

    useEffect(() => {
        if (!data.length) return;
        const width = d3Container.current.clientWidth;
        const height = d3Container.current.clientHeight;
        const margin = { top: 20, right: 20, bottom: 80, left: 60 };
        const mainH = (height - 60) * 0.7;
        const navH = (height - 60) * 0.3;
        d3.select(d3Container.current).selectAll('*').remove();
        const svg = d3.select(d3Container.current)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('background', 'linear-gradient(120deg,#181a1b,#232526)');
        // 主圖區間
        const shown = data.slice(range[0], range[1] + 1);
        // X/Y
        const x = d3.scaleBand().domain(shown.map(d => d.date)).range([margin.left, width - margin.right]).padding(0.3);
        const y = d3.scaleLinear().domain([d3.min(shown, d => d.low), d3.max(shown, d => d.high)]).nice().range([mainH, margin.top]);
        // 主K線圖
        const g = svg.append('g');
        // K棒
        g.selectAll('.kbar').data(shown).enter().append('rect')
            .attr('class', 'kbar')
            .attr('x', d => x(d.date))
            .attr('y', d => y(Math.max(d.open, d.close)))
            .attr('width', x.bandwidth())
            .attr('height', d => Math.max(1, Math.abs(y(d.open) - y(d.close))))
            .attr('fill', d => d.close >= d.open ? '#26ff8a' : '#ff3c3c')
            .attr('stroke', '#222')
            .attr('stroke-width', 1.5);
        g.selectAll('.kline').data(shown).enter().append('line')
            .attr('class', 'kline')
            .attr('x1', d => x(d.date) + x.bandwidth() / 2)
            .attr('x2', d => x(d.date) + x.bandwidth() / 2)
            .attr('y1', d => y(d.high))
            .attr('y2', d => y(d.low))
            .attr('stroke', d => d.close >= d.open ? '#26ff8a' : '#ff3c3c')
            .attr('stroke-width', 2.5)
            .attr('opacity', 0.9);
        // 主圖軸（每半年標記一次）
        const allDates = shown.map(d => d.date);
        const minDate = allDates[0], maxDate = allDates[allDates.length - 1];
        const ticks = [];
        let d = new Date(minDate);
        d.setMonth(d.getMonth() - d.getMonth() % 6, 1);
        while (d <= maxDate) {
            ticks.push(new Date(d));
            d.setMonth(d.getMonth() + 6);
        }
        g.append('g').attr('transform', `translate(0,${mainH})`).call(d3.axisBottom(x)
            .tickValues(ticks.filter(t => x(t) !== undefined))
            .tickFormat(d3.timeFormat('%Y-%m'))).selectAll('text').attr('fill', '#fff').attr('transform', 'rotate(-30)').style('text-anchor', 'end');
        g.append('g').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(y)).selectAll('text').attr('fill', '#fff');
        // 副圖全局
        const navY = d3.scaleLinear().domain([d3.min(data, d => d.low), d3.max(data, d => d.high)]).nice().range([height - margin.bottom, height - navH]);
        const navX = d3.scaleLinear().domain([0, data.length - 1]).range([margin.left, width - margin.right]);
        // 全局主線資料
        const navLine = d3.line()
            .x((d, i) => navX(i))
            .y(d => navY(d.close));
        // 填色區域
        const navArea = d3.area()
            .x((d, i) => navX(i))
            .y0(navY.range()[0])
            .y1(d => navY(d.close));
        const nav = svg.append('g');

        // 全局K線區域底色（白底） - 先繪製背景 (用戶要求移除此白框)
        // nav.append('rect')
        //     .attr('x', margin.left)
        //     .attr('y', height - navH)
        //     .attr('width', width - margin.left - margin.right)
        //     .attr('height', navH)
        //     .attr('fill', '#ffffff')
        //     .attr('opacity', 0.9)
        //     .attr('stroke', '#ddd')
        //     .attr('stroke-width', 1);

        nav.append('path')
            .datum(data)
            .attr('d', navArea)
            .attr('fill', '#1976d2aa');
        nav.append('path')
            .datum(data)
            .attr('d', navLine)
            .attr('stroke', '#1976d2')
            .attr('stroke-width', 2.5)
            .attr('fill', 'none');
        nav.selectAll('.navkbar').data(data).enter().append('rect')
            .attr('class', 'navkbar')
            .attr('x', (d, i) => navX(i) - 0.5)
            .attr('y', d => navY(Math.max(d.open, d.close)))
            .attr('width', 1)
            .attr('height', d => Math.max(1, Math.abs(navY(d.open) - navY(d.close))))
            .attr('fill', d => d.close >= d.open ? '#26ff8a' : '#ff3c3c')
            .attr('opacity', 0.18);
        // 框選區間
        let brush = d3.brushX()
            .extent([[margin.left, height - navH], [width - margin.right, height - margin.bottom]])
            .on('brush end', (event) => {
                // 取得目前選取範圍（拖曳中或結束）
                const sel = event.selection;
                nav.selectAll('.mask-selected').remove();
                if (sel) {
                    nav.append('rect')
                        .attr('class', 'mask-selected')
                        .attr('x', sel[0])
                        .attr('y', height - navH)
                        .attr('width', sel[1] - sel[0])
                        .attr('height', navH - (height - margin.bottom))
                        .attr('fill', 'rgba(100, 50, 200, 0.18)');
                }
                // end事件才setRange
                if (event.type === 'end' && sel) {
                    const [x0, x1] = sel;
                    const idx0 = Math.max(0, Math.floor((x0 - margin.left) / (width - margin.left - margin.right) * data.length));
                    const idx1 = Math.min(data.length - 1, Math.ceil((x1 - margin.left) / (width - margin.left - margin.right) * data.length));
                    setRange([idx0, idx1]);
                    if (onRangeChange) onRangeChange([idx0, idx1]);
                }
            });
        nav.append('g').call(brush).call(g => g.select('.overlay').attr('cursor', 'crosshair'));
        // 全局X軸（每半年標記一次）
        nav.selectAll('.nav-xaxis').remove();
        const navTicks = [];
        let nd = new Date(data[0].date);
        nd.setMonth(nd.getMonth() - nd.getMonth() % 6, 1);
        while (nd <= data[data.length - 1].date) {
            navTicks.push(new Date(nd));
            nd.setMonth(nd.getMonth() + 6);
        }
        nav.append('g')
            .attr('class', 'nav-xaxis')
            .call(d3.axisBottom(d3.scaleBand().domain(data.map((d, i) => i)).range([margin.left, width - margin.right])
            ).tickValues(navTicks.map(t => data.findIndex(d => d3.timeFormat('%Y-%m')(d.date) === d3.timeFormat('%Y-%m')(t))).filter(idx => idx >= 0))
                .tickFormat(idx => d3.timeFormat('%Y-%m')(data[idx].date)))
            .attr('transform', `translate(0,${height - margin.bottom})`)
            .selectAll('text').attr('fill', '#fff').attr('transform', 'rotate(-30)').style('text-anchor', 'end');

        // 最終的選取範圍高亮框（用之前有效的方式）
        const rangeStart = Math.min(range[0], range[1]);
        const rangeEnd = Math.max(range[0], range[1]);
        const x0 = navX(rangeStart);
        const x1 = navX(rangeEnd);

        // 綠色 range 框（之前確認有效的方式）
        nav.append('rect')
            .attr('class', 'range-box')
            .attr('x', x0)
            .attr('y', height - navH + 29)
            .attr('width', Math.max(10, x1 - x0))
            .attr('height', 20)
            .attr('fill', 'lime')
            .attr('stroke', 'red')
            .attr('stroke-width', 3)
            .attr('opacity', 0.5);
    }, [data, range]);

    return (
        <div style={{ width: '100%', height: '100%', background: 'none' }}>
            <div ref={d3Container} style={{ width: '100%', height: '500px', borderRadius: '12px', position: 'relative' }} />
        </div>
    );
}

export default KLineChart; 