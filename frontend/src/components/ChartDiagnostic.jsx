import React, { useEffect, useState } from 'react';

const ChartDiagnostic = () => {
    const [diagnosticResult, setDiagnosticResult] = useState({});

    useEffect(() => {
        const runDiagnostic = () => {
            const result = {};

            // 1. 檢查 ECharts 容器
            const echartsContainer = document.querySelector('.echarts-for-react, [data-echarts]');
            result.echartsContainer = {
                exists: !!echartsContainer,
                element: echartsContainer,
                styles: echartsContainer ? getComputedStyle(echartsContainer) : null,
                rect: echartsContainer ? echartsContainer.getBoundingClientRect() : null
            };

            // 2. 檢查 Canvas 元素
            const canvas = document.querySelector('canvas');
            result.canvas = {
                exists: !!canvas,
                element: canvas,
                styles: canvas ? getComputedStyle(canvas) : null,
                rect: canvas ? canvas.getBoundingClientRect() : null,
                context: canvas ? canvas.getContext('2d') : null
            };

            // 3. 檢查父層容器
            if (echartsContainer) {
                let parent = echartsContainer.parentElement;
                const parents = [];
                while (parent && parent !== document.body) {
                    const styles = getComputedStyle(parent);
                    parents.push({
                        tagName: parent.tagName,
                        className: parent.className,
                        rect: parent.getBoundingClientRect(),
                        display: styles.display,
                        height: styles.height,
                        overflow: styles.overflow,
                        visibility: styles.visibility,
                        opacity: styles.opacity,
                        zIndex: styles.zIndex
                    });
                    parent = parent.parentElement;
                }
                result.parentChain = parents;
            }

            // 4. 檢查 ECharts 實例
            if (window.echarts) {
                result.echarts = {
                    version: window.echarts.version || 'unknown',
                    instances: window.echarts.getInstanceByDom ? 
                        (echartsContainer ? window.echarts.getInstanceByDom(echartsContainer) : null) : null
                };
            }

            setDiagnosticResult(result);
        };

        // 延遲執行，確保組件都已渲染
        const timer = setTimeout(runDiagnostic, 1000);
        return () => clearTimeout(timer);
    }, []);

    const renderDiagnostic = () => {
        const { echartsContainer, canvas, parentChain, echarts } = diagnosticResult;

        return (
            <div style={{ 
                padding: '20px', 
                backgroundColor: '#f6f8fa', 
                border: '1px solid #e1e4e8',
                borderRadius: '8px',
                margin: '20px',
                fontFamily: 'monospace',
                fontSize: '12px'
            }}>
                <h3 style={{ color: '#24292e', marginBottom: '16px' }}>🔍 ECharts 診斷報告</h3>
                
                {/* ECharts 容器檢查 */}
                <div style={{ marginBottom: '16px' }}>
                    <h4 style={{ color: echartsContainer?.exists ? '#28a745' : '#d73a49' }}>
                        1. ECharts 容器: {echartsContainer?.exists ? '✅ 存在' : '❌ 不存在'}
                    </h4>
                    {echartsContainer?.exists && (
                        <div style={{ marginLeft: '16px', color: '#586069' }}>
                            <p>尺寸: {echartsContainer.rect?.width} x {echartsContainer.rect?.height}</p>
                            <p>Display: {echartsContainer.styles?.display}</p>
                            <p>Height: {echartsContainer.styles?.height}</p>
                            <p>Visibility: {echartsContainer.styles?.visibility}</p>
                            <p>Opacity: {echartsContainer.styles?.opacity}</p>
                        </div>
                    )}
                </div>

                {/* Canvas 檢查 */}
                <div style={{ marginBottom: '16px' }}>
                    <h4 style={{ color: canvas?.exists ? '#28a745' : '#d73a49' }}>
                        2. Canvas 元素: {canvas?.exists ? '✅ 存在' : '❌ 不存在'}
                    </h4>
                    {canvas?.exists && (
                        <div style={{ marginLeft: '16px', color: '#586069' }}>
                            <p>尺寸: {canvas.rect?.width} x {canvas.rect?.height}</p>
                            <p>Display: {canvas.styles?.display}</p>
                            <p>Visibility: {canvas.styles?.visibility}</p>
                            <p>Context: {canvas.context ? '有效' : '無效'}</p>
                        </div>
                    )}
                </div>

                {/* 父層容器檢查 */}
                {parentChain && (
                    <div style={{ marginBottom: '16px' }}>
                        <h4 style={{ color: '#24292e' }}>3. 父層容器鏈</h4>
                        {parentChain.map((parent, index) => (
                            <div key={index} style={{ 
                                marginLeft: `${16 + index * 8}px`, 
                                color: '#586069',
                                marginBottom: '4px'
                            }}>
                                <span style={{ fontWeight: 'bold' }}>{parent.tagName}</span>
                                {parent.className && <span>.{parent.className}</span>}
                                <span> - Display: {parent.display}, Height: {parent.height}</span>
                                {parent.rect.height === 0 && 
                                    <span style={{ color: '#d73a49', fontWeight: 'bold' }}> ⚠️ 高度為 0</span>
                                }
                            </div>
                        ))}
                    </div>
                )}

                {/* ECharts 實例檢查 */}
                {echarts && (
                    <div style={{ marginBottom: '16px' }}>
                        <h4 style={{ color: '#24292e' }}>4. ECharts 實例</h4>
                        <div style={{ marginLeft: '16px', color: '#586069' }}>
                            <p>版本: {echarts.version}</p>
                            <p>實例: {echarts.instances ? '有效' : '無效'}</p>
                        </div>
                    </div>
                )}

                {/* 修復建議 */}
                <div style={{ marginTop: '20px', padding: '12px', backgroundColor: '#fff3cd', border: '1px solid #ffeaa7', borderRadius: '4px' }}>
                    <h4 style={{ color: '#856404', marginBottom: '8px' }}>🔧 快速修復</h4>
                    <button 
                        onClick={() => {
                            const c = document.querySelector('.echarts-for-react, [data-echarts]');
                            if (c) {
                                c.style.height = '350px';
                                c.style.width = '100%';
                                c.style.display = 'block';
                                c.style.visibility = 'visible';
                                c.style.opacity = '1';
                            }
                            const cv = document.querySelector('canvas');
                            if (cv) {
                                cv.style.background = '#f9f9f9';
                                cv.style.border = '1px solid #ccc';
                            }
                        }}
                        style={{
                            padding: '8px 16px',
                            backgroundColor: '#007bff',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer'
                        }}
                    >
                        強制顯示圖表
                    </button>
                </div>
            </div>
        );
    };

    return (
        <div>
            {Object.keys(diagnosticResult).length > 0 ? renderDiagnostic() : (
                <div style={{ padding: '20px', textAlign: 'center' }}>
                    <div>🔍 正在診斷...</div>
                </div>
            )}
        </div>
    );
};

export default ChartDiagnostic; 