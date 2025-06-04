import React from 'react';

function ProjectIntro() {
    return (
        <div className="tab-content" style={{ padding: '2rem', color: '#eee', maxWidth: '800px', margin: '2rem auto', textAlign: 'left', background: 'rgba(30, 30, 50, 0.7)', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
            <h2 style={{ textAlign: 'center', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255, 255, 255, 0.2)', paddingBottom: '1rem' }}>專案介紹</h2>
            <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>此專案旨在構建一個比特幣情緒分析視覺化系統。</p>
            <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>系統結合了以下幾個主要部分來提供全面的市場洞察：</p>
            <ul style={{ listStylePosition: 'inside', paddingLeft: '1rem', marginBottom: '1rem' }}>
                <li style={{ marginBottom: '0.5rem' }}><strong>價格趨勢分析:</strong> 展示詳細的比特幣K線圖，幫助用戶理解歷史價格變動、開盤價、收盤價、最高價和最低價。</li>
                <li style={{ marginBottom: '0.5rem' }}><strong>不確定性因子分析 (UASTL):</strong> 透過UASTL模型分解時間序列數據，揭示季節性模式、長期趨勢和隨機殘差，輔助判斷市場波動性。</li>
                <li style={{ marginBottom: '0.5rem' }}><strong>社群情緒分析 (未來集成):</strong> 計劃集成社交媒體情緒數據，量化市場參與者的整體情緒傾向，並將其與價格及不確定性數據結合分析。</li>
            </ul>
            <p style={{ marginBottom: '1rem', lineHeight: '1.6' }}>使用者可以通過切換不同的圖表和指標來觀察市場動態、潛在的情緒影響，以及各種分析模型提供的洞見。</p>
            <p style={{ lineHeight: '1.6' }}>本系統致力於提供一個直觀、易用的界面，幫助用戶做出更明智的決策。</p>

        </div>
    );
}

export default ProjectIntro; 