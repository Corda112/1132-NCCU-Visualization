import React, { useState } from 'react';
import ClusterBubbleChart from './ClusterBubbleChart';
import ClusterTimeline from './ClusterTimeline';
import ClusterHeatmap from './ClusterHeatmap';
import ClusterWordCloud from './ClusterWordCloud';

function ClusterDashboard({ range }) {
    const [selected, setSelected] = useState(null);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <ClusterBubbleChart range={range} onSelect={setSelected} />
            <ClusterTimeline range={range} onBrush={() => {}} />
            <ClusterHeatmap range={range} />
            <ClusterWordCloud clusterId={selected} range={range} />
        </div>
    );
}

export default ClusterDashboard;
