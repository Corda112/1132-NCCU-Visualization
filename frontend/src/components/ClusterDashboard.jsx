import React, { useState } from 'react';
import ClusterBubbleChart from './ClusterBubbleChart';
import ClusterTimeline from './ClusterTimeline';
import ClusterHeatmap from './ClusterHeatmap';
import ClusterWordCloud from './ClusterWordCloud';

function ClusterDashboard({ range, onTermSelect }) {
    const [selected, setSelected] = useState(null);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <ClusterBubbleChart range={range} selectedCluster={selected} onSelect={setSelected} />
            <ClusterTimeline range={range} selectedCluster={selected} onBrush={() => {}} />
            <ClusterHeatmap range={range} selectedCluster={selected} />
            <ClusterWordCloud clusterId={selected} range={range} onTermClick={(term) => onTermSelect && onTermSelect({ term })} />
        </div>
    );
}

export default ClusterDashboard;
