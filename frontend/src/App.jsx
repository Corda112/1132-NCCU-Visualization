import React, { useState, useCallback } from 'react';
import './App.css';
import KLineChart from './components/KLineChart';
import UASTLChart from './components/UASTLChart';
import ProjectIntro from './components/ProjectIntro'; // 引入 ProjectIntro 組件

function App() {
    const [selectedRange, setSelectedRange] = useState(null);
    const [activeTab, setActiveTab] = useState('intro'); // 'intro' 或 'charts'

    const handleRangeChange = useCallback((range) => {
        setSelectedRange(range);
    }, []);

    const renderContent = () => {
        if (activeTab === 'intro') {
            return <ProjectIntro />; // 使用 ProjectIntro 組件
        } else if (activeTab === 'charts') {
            return (
                <main className="main-dashboard">
                    {/* Left Panel - Charts */}
                    <div className="charts-panel">
                        {/* Price Analysis Section */}
                        <section className="price-section">
                            <div className="section-header">
                                <h2 className="section-title">
                                    <span className="title-icon">📈</span>
                                    價格趨勢分析
                                </h2>
                                <div className="chart-controls">
                                    <button className="control-btn active">K線圖</button>
                                </div>
                            </div>
                            <div className="kline-chart-container">
                                <KLineChart onRangeChange={handleRangeChange} />
                            </div>
                        </section>

                        {/* Uncertainty Analysis Section */}
                        <section className="uncertainty-section">
                            <div className="section-header">
                                <h2 className="section-title">
                                    <span className="title-icon">🎯</span>
                                    不確定性因子分析 (UASTL)
                                </h2>
                                <div className="uncertainty-info">
                                    <span className="info-badge">季節性分解</span>
                                    <span className="info-badge">趨勢預測</span>
                                    <span className="info-badge">殘差分析</span>
                                </div>
                            </div>
                            <div className="uastl-chart-container">
                                <UASTLChart range={selectedRange} />
                            </div>
                        </section>
                    </div>

                    {/* Right Panel - Analysis & Insights */}
                    <aside className="insights-panel">
                        {/* Sentiment Analysis Section */}
                        <section className="sentiment-section-sidebar">
                            <div className="section-header">
                                <h2 className="section-title">
                                    <span className="title-icon">📊</span>
                                    社群情緒分析
                                </h2>
                            </div>
                            <div className="sentiment-chart-sidebar">
                                <span>內容待添加</span>
                            </div>
                        </section>

                        <div className="panel-header">
                            <h3>分析洞察</h3>
                        </div>

                        <div className="insight-card">
                            <h4>市場情緒指標</h4>
                            <span>內容待添加</span>
                        </div>

                        <div className="insight-card">
                            <h4>價格預測</h4>
                            <span>內容待添加</span>
                        </div>

                        <div className="insight-card">
                            <h4>關鍵指標</h4>
                            <span>內容待添加</span>
                        </div>

                        <div className="insight-card">
                            <h4>最新消息影響</h4>
                            <span>內容待添加</span>
                        </div>
                    </aside>
                </main>
            );
        }
    };

    return (
        <div className="app-container">
            <header className="header-section">
                <div className="header-content">
                    <div className="logo-area">
                        <div className="bitcoin-icon">₿</div>
                        <h1 className="main-title">Bitcoin 情緒分析視覺化系統</h1>
                    </div>

                    {/* 新增的頁籤區域 */}
                    <nav className="tabs-navigation">
                        <button
                            className={`tab-button ${activeTab === 'intro' ? 'active' : ''}`}
                            onClick={() => setActiveTab('intro')}
                        >
                            專案介紹
                        </button>
                        <button
                            className={`tab-button ${activeTab === 'charts' ? 'active' : ''}`}
                            onClick={() => setActiveTab('charts')}
                        >
                            K線 & UASTL
                        </button>
                    </nav>

                    <div className="header-stats">
                        <div className="stat-item">
                            <span className="stat-label">數據範圍</span>
                            <span className="stat-value">2020-2024</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">數據類型</span>
                            <span className="stat-value">歷史數據</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">置信度</span>
                            <span className="stat-value">99%</span>
                        </div>
                    </div>
                </div>
            </header>

            {renderContent()} {/* 根據 activeTab 渲染內容 */}
        </div>
    );
}

export default App; 