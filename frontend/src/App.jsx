import React, { useState, useCallback, useEffect } from 'react';
import './App.css';
import KLineChart from './components/KLineChart';
import UASTLChart from './components/UASTLChart';
import ProjectIntro from './components/ProjectIntro';
import RightPanel from './components/RightPanel';

function App() {
    const [selectedRange, setSelectedRange] = useState(null);
    const [activeTab, setActiveTab] = useState('charts'); // 'intro' or 'charts'
    const [rightPanelTab, setRightPanelTab] = useState('sentiment'); // sentiment, term, ngram, cluster, reading_pane

    // Data states
    const [loading, setLoading] = useState(true);
    const [termFreqData, setTermFreqData] = useState(null);
    const [semanticData, setSemanticData] = useState(null);
    const [readingPaneData, setReadingPaneData] = useState(null);
    const [error, setError] = useState(null); // Add error state

    // Filter states for Reading Pane
    const [filters, setFilters] = useState({
        term: '',
        ngram: '',
        cluster_id: '',
        semantic_query: ''
    });

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null); // Reset error state
                const [termRes, semanticRes, readingRes] = await Promise.all([
                    fetch('/data/term_ngram_frequency.json'),
                    fetch('/data/semantic_clustering_sentiment.json'),
                    fetch('/data/reading_pane_data.json')
                ]);

                // Check for HTTP errors
                if (!termRes.ok || !semanticRes.ok || !readingRes.ok) {
                    throw new Error(`HTTP error! status: ${termRes.status} ${semanticRes.status} ${readingRes.status}`);
                }

                const termData = await termRes.json();
                const semanticData = await semanticRes.json();
                const readingData = await readingRes.json();

                setTermFreqData(termData);
                setSemanticData(semanticData);
                setReadingPaneData(readingData);

            } catch (error) {
                console.error("Failed to fetch data:", error);
                setError(error.message); // Set error state
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const handleRangeChange = useCallback((range) => {
        setSelectedRange(range);
    }, []);

    const renderContent = () => {
        if (activeTab === 'intro') {
            return <ProjectIntro />;
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
                        <RightPanel
                            activeTab={rightPanelTab}
                            setActiveTab={setRightPanelTab}
                            loading={loading}
                            error={error} // Pass error state down
                            selectedRange={selectedRange}
                            termFreqData={termFreqData}
                            semanticData={semanticData}
                            readingPaneData={readingPaneData}
                            filters={filters}
                            setFilters={setFilters}
                        />
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

                    {/* Tabs Navigation */}
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
                            儀表板
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

            {renderContent()}
        </div>
    );
}

export default App; 