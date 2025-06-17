import React, { useEffect, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

function ClusterWordCloud({ clusterId, range, onTermClick }) {
    const [words, setWords] = useState([]);

    useEffect(() => {
        if (clusterId === null || clusterId === undefined || !range) {
            setWords([]);
            return;
        }
        const startDate = new Date(range.from).toISOString().split('T')[0];
        const endDate = new Date(range.to).toISOString().split('T')[0];
        axios.get('http://localhost:3001/api/clusters', { params: { startDate, endDate } })
            .then(res => {
                const texts = res.data.filter(d => d.cluster_id === clusterId).map(d => d.cleaned_text.toLowerCase());
                const freq = {};
                texts.forEach(t => {
                    t.split(/[^a-zA-Z0-9]+/).forEach(w => {
                        if (w.length > 2) {
                            freq[w] = (freq[w] || 0) + 1;
                        }
                    });
                });
                const items = Object.entries(freq).sort((a,b) => b[1]-a[1]).slice(0,40).map(([text,size])=>({text,size}));
                setWords(items);
            })
            .catch(err => console.error('Wordcloud fetch error', err));
    }, [clusterId, range]);

    if (!words.length) return <div style={{height:'200px'}}>Select a cluster to see keywords</div>;

    const max = d3.max(words, d => d.size) || 1;

    return (
        <div style={{ display: 'flex', flexWrap: 'wrap', width: '100%', height: '200px' }}>
            {words.map((w,i) => (
                <span key={i}
                    onClick={() => onTermClick && onTermClick(w.text)}
                    style={{
                        fontSize: `${10 + (w.size / max) * 30}px`,
                        marginRight: '8px',
                        cursor: 'pointer'
                    }}>
                    {w.text}
                </span>
            ))}
        </div>
    );
}

export default ClusterWordCloud;
