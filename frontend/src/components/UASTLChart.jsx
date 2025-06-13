import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

// Component colors based on user's reference image (不確定性圖.jpg)
const componentColors = {
    input: "#483D8B",     // DarkSlateBlue for Input
    trend: "#8B0000",     // DarkRed for Trend
    seasonal1: "#008080", // Teal for Seasonal 1
    seasonal2: "#8A2BE2", // BlueViolet for Seasonal 2
    seasonal3: "#BDB76B", // DarkKhaki (olive-like) for Seasonal 3
    residual: "#708090"   // SlateGray for Residual
};

// Sample line colors inspired by MATLAB's initialColorOrder
const sampleLineColors = ['#EDAF20', '#77AB30', '#7E2F8E', '#A2142F', '#D95319', '#0072BD', '#4DBEEE'];

// Sigma levels for 50% and 99% CI (from MATLAB plot_dist.m)
const sigmaLevels = {
    '50%': 0.674490,
    '99%': 2.575829
};

// CI fill colors based on MATLAB's approach (blue channel, lighter for wider CI)
const baseMatlabCIBlue = d3.rgb(55, 126, 184);
const ciFillColor99 = d3.interpolateRgb(baseMatlabCIBlue, "white")(0.5).toString(); // Corresponds to MATLAB's col(3,:), 50% mix with white
const ciFillColor50 = d3.interpolateRgb(baseMatlabCIBlue, "white")(0.25).toString(); // Corresponds to MATLAB's col(2,:), 25% mix with white
const ciFillOpacity = 0.5; // General opacity for CIs, similar to MATLAB's FaceAlpha

function UASTLChart({ range }) {
    const svgRef = useRef();

    useEffect(() => {
        if (!svgRef.current || !svgRef.current.parentNode) return;

        axios.get('http://localhost:3001/api/uastl').then(res => {
            let rawData = res.data;
            if (!Array.isArray(rawData) || rawData.length === 0) {
                console.error("No data or invalid data format received from API.");
                // Display error message in SVG
                const tempSvg = d3.select(svgRef.current);
                tempSvg.selectAll("*").remove();
                tempSvg.append("text")
                    .attr("x", tempSvg.attr("width") / 2 || 200)
                    .attr("y", tempSvg.attr("height") / 2 || 100)
                    .attr("text-anchor", "middle")
                    .text("No data available to display.");
                return;
            }

            const dataWithDates = rawData.map(d => ({ ...d, dateObj: new Date(d.date) }));
            let displayData = dataWithDates;

            if (range && range.from && range.to) {
                displayData = dataWithDates.filter(d => {
                    const time = d.dateObj.getTime();
                    return time >= range.from && time <= range.to;
                });
            }

            if (displayData.length === 0) return;

            const parentNode = svgRef.current.parentNode;
            const containerWidth = parentNode.clientWidth;
            const containerHeight = parentNode.clientHeight;

            d3.select(svgRef.current).selectAll("*").remove();
            const svg = d3.select(svgRef.current)
                .attr("width", containerWidth)
                .attr("height", containerHeight);

            const firstDataPoint = displayData[0];
            const coPoint = firstDataPoint.coPoint_used ? +firstDataPoint.coPoint_used - 1 : null;
            const p_used_str = firstDataPoint.p_used || "7,90,365";
            const p_used = p_used_str.split(',').map(Number);
            const numSamples = firstDataPoint.num_samples_added || 3;

            const seasonalKeyBase = 'seasonal';
            let componentDefinitions = [
                { key: 'input', name: 'Input Data', dataKey: 'input', pValue: null },
                { key: 'trend', name: 'Trend Component', dataKey: 'trend', pValue: null },
            ];
            for (let i = 0; i < p_used.length; i++) {
                const sNum = i + 1;
                if (firstDataPoint.hasOwnProperty(`${seasonalKeyBase}${sNum}_mean`)) {
                    componentDefinitions.push({
                        key: `${seasonalKeyBase}${sNum}`,
                        name: `Seasonal Component ${sNum} (p=${p_used[i]})`,
                        dataKey: `${seasonalKeyBase}${sNum}`,
                        pValue: p_used[i]
                    });
                }
            }
            componentDefinitions.push({ key: 'residual', name: 'Residual Component', dataKey: 'residual', pValue: null });

            const processedComponents = componentDefinitions.map(compDef => {
                return {
                    ...compDef,
                    plotData: displayData.map(d => {
                        const item = {
                            dateObj: d.dateObj,
                            mean: +d[`${compDef.dataKey}_mean`],
                            std: +d[`${compDef.dataKey}_std`],
                            corr: +d[`corr_${compDef.dataKey}`]
                        };
                        for (let s = 1; s <= numSamples; s++) {
                            item[`sample${s}`] = +d[`sample${s}_${compDef.dataKey}`];
                        }
                        return item;
                    })
                };
            });

            const margin = { top: 50, right: 30, bottom: 60, left: 60 };
            const numComponents = processedComponents.length;
            const totalSubplotVerticalSpace = containerHeight - margin.top - margin.bottom;
            const subplotSpacing = 25;
            const correlationStripHeight = 25;

            const singleSubplotAllocatedHeight = (totalSubplotVerticalSpace / numComponents) - subplotSpacing;
            const drawableSubplotHeight = singleSubplotAllocatedHeight - correlationStripHeight;

            const xScale = d3.scaleTime()
                .domain(d3.extent(displayData, d => d.dateObj))
                .range([margin.left, containerWidth - margin.right]);

            const coolWarmColorScale = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);

            svg.append("text")
                .attr("x", containerWidth / 2)
                .attr("y", margin.top / 2.5)
                .attr("text-anchor", "middle")
                .style("font-size", "16px")
                .style("font-weight", "bold")
                .text("Uncertainty-Aware Seasonal-Trend Decomposition");

            processedComponents.forEach((component, i) => {
                const subplotGroupYOffset = margin.top + i * (singleSubplotAllocatedHeight + subplotSpacing);
                const mainPlotAreaY = subplotGroupYOffset;
                const corrAreaY = mainPlotAreaY + drawableSubplotHeight + 5; // 5px gap

                const allYValues = [];
                component.plotData.forEach(d => {
                    if (!isNaN(d.mean) && !isNaN(d.std)) {
                        allYValues.push(d.mean - sigmaLevels['99%'] * d.std);
                        allYValues.push(d.mean + sigmaLevels['99%'] * d.std);
                    } else if (!isNaN(d.mean)) {
                        allYValues.push(d.mean);
                    }
                    for (let s = 1; s <= numSamples; s++) {
                        if (!isNaN(d[`sample${s}`])) allYValues.push(d[`sample${s}`]);
                    }
                });

                let yMin = d3.min(allYValues);
                let yMax = d3.max(allYValues);

                if (allYValues.length === 0 || yMin === yMax) {
                    yMin = (yMin === undefined) ? -1 : yMin - 0.5;
                    yMax = (yMax === undefined) ? 1 : yMax + 0.5;
                }
                const yPadding = (yMax - yMin) * 0.1 || 0.1;
                yMin -= yPadding;
                yMax += yPadding;

                const yScale = d3.scaleLinear()
                    .domain([yMin, yMax])
                    .range([mainPlotAreaY + drawableSubplotHeight, mainPlotAreaY]);

                const subPlotGroup = svg.append("g").attr("class", `subplot-${component.key}`);

                subPlotGroup.append("rect")
                    .attr("x", margin.left)
                    .attr("y", mainPlotAreaY)
                    .attr("width", containerWidth - margin.left - margin.right)
                    .attr("height", drawableSubplotHeight)
                    .attr("fill", "#f9f9f9")
                    .attr("stroke", "#e0e0e0");

                const area99 = d3.area()
                    .x(d => xScale(d.dateObj))
                    .y0(d => yScale(d.mean - sigmaLevels['99%'] * d.std))
                    .y1(d => yScale(d.mean + sigmaLevels['99%'] * d.std))
                    .defined(d => !isNaN(d.mean) && !isNaN(d.std) && d.std > 0);

                subPlotGroup.append("path")
                    .datum(component.plotData)
                    .attr("fill", ciFillColor99)
                    .attr("opacity", ciFillOpacity)
                    .attr("d", area99);

                const area50 = d3.area()
                    .x(d => xScale(d.dateObj))
                    .y0(d => yScale(d.mean - sigmaLevels['50%'] * d.std))
                    .y1(d => yScale(d.mean + sigmaLevels['50%'] * d.std))
                    .defined(d => !isNaN(d.mean) && !isNaN(d.std) && d.std > 0);

                subPlotGroup.append("path")
                    .datum(component.plotData)
                    .attr("fill", ciFillColor50)
                    .attr("opacity", ciFillOpacity)
                    .attr("d", area50);

                for (let s = 1; s <= numSamples; s++) {
                    const sampleKey = `sample${s}`;
                    const line = d3.line()
                        .x(d => xScale(d.dateObj))
                        .y(d => yScale(d[sampleKey]))
                        .defined(d => !isNaN(d[sampleKey]));

                    subPlotGroup.append("path")
                        .datum(component.plotData)
                        .attr("fill", "none")
                        .attr("stroke", sampleLineColors[(s - 1) % sampleLineColors.length])
                        .attr("stroke-width", 1)
                        .attr("opacity", 0.7)
                        .attr("d", line);
                }

                const meanLine = d3.line()
                    .x(d => xScale(d.dateObj))
                    .y(d => yScale(d.mean))
                    .defined(d => !isNaN(d.mean));

                subPlotGroup.append("path")
                    .datum(component.plotData)
                    .attr("fill", "none")
                    .attr("stroke", componentColors['residual'])
                    .attr("stroke-width", 1.5)
                    .attr("opacity", 0.95)
                    .attr("d", meanLine);

                subPlotGroup.append("g")
                    .attr("transform", `translate(${margin.left},0)`)
                    .call(d3.axisLeft(yScale).ticks(4).tickSizeOuter(0))
                    .selectAll("text").style("font-size", "9px");

                const maxAbsCorrForComponent = d3.max(component.plotData, d => Math.abs(d.corr)) || 1;
                component.plotData.forEach((d, idx) => {
                    if (isNaN(d.corr)) return;

                    const barX = xScale(d.dateObj);
                    let barWidth = 2; // Default width
                    if (idx < displayData.length - 1) {
                        barWidth = xScale(displayData[idx + 1].dateObj) - barX;
                    } else if (idx > 0) {
                        barWidth = barX - xScale(displayData[idx - 1].dateObj);
                    }

                    const barHeightRatio = Math.abs(d.corr) / maxAbsCorrForComponent;
                    const visualBarHeight = barHeightRatio * (correlationStripHeight / 2);

                    let rectY;
                    if (d.corr >= 0) {
                        rectY = corrAreaY + (correlationStripHeight / 2) - visualBarHeight;
                    } else {
                        rectY = corrAreaY + (correlationStripHeight / 2);
                    }

                    subPlotGroup.append("rect")
                        .attr("x", barX - barWidth / 2 + 0.5)
                        .attr("y", rectY)
                        .attr("width", Math.max(1, barWidth - 1))
                        .attr("height", Math.max(0.5, visualBarHeight))
                        .attr("fill", d3.rgb(coolWarmColorScale(d.corr)).darker(0.2).toString())
                        .attr("stroke", "none");
                });

                if (i === numComponents - 1) {
                    const lastSubplotXAxisLineY = containerHeight - margin.bottom;
                    svg.append("g")
                        .attr("transform", `translate(0,${lastSubplotXAxisLineY})`)
                        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.timeFormat("%Y-%m-%d")))
                        .selectAll("text").style("font-size", "9px")
                        .attr("transform", "rotate(-45)")
                        .style("text-anchor", "end");

                    svg.append("text")
                        .attr("x", containerWidth / 2)
                        .attr("y", containerHeight - 10)
                        .attr("text-anchor", "middle")
                        .style("font-size", "11px")
                        .text("Date");
                }

                subPlotGroup.append("text")
                    .attr("x", containerWidth / 2)
                    .attr("y", mainPlotAreaY - 7)
                    .attr("text-anchor", "middle")
                    .style("font-size", "11px")
                    .style("font-weight", "600")
                    .style("fill", componentColors[component.key] || "black")
                    .text(component.name);

                if (component.key !== 'input' && yMin <= 0 && yMax >= 0) {
                    subPlotGroup.append("line")
                        .attr("x1", margin.left)
                        .attr("x2", containerWidth - margin.right)
                        .attr("y1", yScale(0))
                        .attr("y2", yScale(0))
                        .attr("stroke", "grey")
                        .attr("stroke-width", 0.7)
                        .attr("stroke-dasharray", "3,3");
                }

                if (coPoint !== null && coPoint >= 0 && coPoint < displayData.length) {
                    const coPointX = xScale(displayData[coPoint].dateObj);
                    if (coPointX >= margin.left && coPointX <= containerWidth - margin.right) {
                        subPlotGroup.append("line")
                            .attr("x1", coPointX)
                            .attr("x2", coPointX)
                            .attr("y1", mainPlotAreaY)
                            .attr("y2", corrAreaY + correlationStripHeight)
                            .attr("stroke", "black")
                            .attr("stroke-width", 1.2);
                    }
                }
            });

        }).catch(error => {
            console.error("Error fetching or processing UASTL data:", error);
            const tempSvg = d3.select(svgRef.current);
            tempSvg.selectAll("*").remove();

            // Safely get container dimensions or use defaults
            let CWidth = 200; // Default width
            let CHeight = 100; // Default height
            if (svgRef.current && svgRef.current.parentNode) {
                const parentNode = svgRef.current.parentNode;
                CWidth = parentNode.clientWidth || CWidth;
                CHeight = parentNode.clientHeight || CHeight;
            }

            tempSvg.append("text")
                .attr("x", CWidth / 2)
                .attr("y", CHeight / 2)
                .attr("text-anchor", "middle")
                .style("font-size", "12px")
                .text("Error loading data. Please check console for details.");
        });

    }, [range]);

    return (
        <div style={{ width: '100%', height: '100%', overflow: 'hidden', background: '#fff' }}>
            <svg ref={svgRef} width="100%" height="100%" style={{ display: 'block' }} />
        </div>
    );
}

export default UASTLChart; 