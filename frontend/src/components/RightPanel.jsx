import React from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    ScatterChart, Scatter, ZAxis
} from 'recharts';

// Import child components
import SentimentChart from './SentimentChart';
import TermChart from './TermChart';
import NgramChart from './NgramChart';
import ClusterChart from './ClusterChart';
import ReadingPane from './ReadingPane';


const RightPanel = ({ activeTab, setActiveTab, loading, error, selectedRange, termFreqData, semanticData, readingPaneData, filters, setFilters }) => {

    const renderTabContent = () => {
        if (loading) {
            return <div className="loading-placeholder">讀取資料中...</div>;
        }

        if (error) {
            return <div className="loading-placeholder error-message">資料載入失敗: {error}<br />請檢查 public/data 資料夾中的檔案是否存在且路徑正確。</div>;
        }

        switch (activeTab) {
            case 'sentiment':
                return <SentimentChart data={semanticData} range={selectedRange} />;
            case 'term':
                return <TermChart data={termFreqData} range={selectedRange} />;
            case 'ngram':
                return <NgramChart data={termFreqData} range={selectedRange} />;
            case 'cluster':
                return <ClusterChart data={semanticData} range={selectedRange} />;
            case 'reading_pane':
                return <ReadingPane semanticData={semanticData} readingPaneData={readingPaneData} filters={filters} setFilters={setFilters} />;
            default:
                return null;
        }
    };

    return (
        <>
            <div className="right-panel-tabs">
                <button onClick={() => setActiveTab('sentiment')} className={activeTab === 'sentiment' ? 'active' : ''}>社群情緒分析</button>
                <button onClick={() => setActiveTab('term')} className={activeTab === 'term' ? 'active' : ''}>Term</button>
                <button onClick={() => setActiveTab('ngram')} className={activeTab === 'ngram' ? 'active' : ''}>N-gram</button>
                <button onClick={() => setActiveTab('cluster')} className={activeTab === 'cluster' ? 'active' : ''}>Clustering</button>
                <button onClick={() => setActiveTab('reading_pane')} className={activeTab === 'reading_pane' ? 'active' : ''}>Reading Pane</button>
            </div>

            <div className="right-panel-content">
                {renderTabContent()}
            </div>
        </>
    );
};

export default RightPanel; 