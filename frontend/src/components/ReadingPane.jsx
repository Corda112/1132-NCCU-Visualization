import React, { useState, useMemo } from 'react';

const ReadingPane = ({ semanticData, readingPaneData, filters, setFilters }) => {
    const [currentPage, setCurrentPage] = useState(1);
    const [selectedPost, setSelectedPost] = useState(null);
    const POSTS_PER_PAGE = 100;

    const handleFilterChange = (e) => {
        const { name, value } = e.target;
        setFilters(prev => ({ ...prev, [name]: value }));
        setCurrentPage(1); // Reset to first page on filter change
    };

    const filteredData = useMemo(() => {
        if (!semanticData) return [];
        return semanticData.filter(post => {
            const lowerCaseText = post.cleaned_text.toLowerCase();
            const termMatch = !filters.term || lowerCaseText.includes(filters.term.toLowerCase());
            const ngramMatch = !filters.ngram || lowerCaseText.includes(filters.ngram.toLowerCase());
            const clusterMatch = !filters.cluster_id || post.cluster_id == filters.cluster_id;
            const semanticMatch = !filters.semantic_query || lowerCaseText.includes(filters.semantic_query.toLowerCase());

            return termMatch && ngramMatch && clusterMatch && semanticMatch;
        });
    }, [semanticData, filters]);

    const paginatedData = useMemo(() => {
        const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
        return filteredData.slice(startIndex, startIndex + POSTS_PER_PAGE);
    }, [filteredData, currentPage]);

    const totalPages = Math.ceil(filteredData.length / POSTS_PER_PAGE);

    const handlePostClick = (postId) => {
        setSelectedPost(readingPaneData[postId]);
    };

    if (!semanticData || !readingPaneData) return <div className="loading-placeholder">等待文章資料...</div>

    return (
        <div className="reading-pane-container">
            <div className="filters-container">
                <input type="text" name="term" placeholder="Filter by Term..." value={filters.term} onChange={handleFilterChange} />
                <input type="text" name="ngram" placeholder="Filter by N-gram..." value={filters.ngram} onChange={handleFilterChange} />
                <input type="number" name="cluster_id" placeholder="Filter by Cluster ID..." value={filters.cluster_id} onChange={handleFilterChange} />
                <input type="text" name="semantic_query" placeholder="Semantic Search..." value={filters.semantic_query} onChange={handleFilterChange} />
            </div>

            <div className="posts-list">
                {paginatedData.map(post => (
                    <div key={post.id} className="post-item" onClick={() => handlePostClick(post.id)}>
                        <p className="post-text">{post.cleaned_text.substring(0, 200)}...</p>
                        <div className="post-meta">
                            <span>Cluster: {post.cluster_id}</span>
                            <span>Sentiment: {post.sentiment.toFixed(2)}</span>
                            <span>{new Date(post.createdAt).toLocaleDateString()}</span>
                        </div>
                    </div>
                ))}
                {paginatedData.length === 0 && <div className="loading-placeholder">無符合篩選條件的文章</div>}
            </div>

            {totalPages > 1 && (
                <div className="pagination">
                    {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                        <button key={page} onClick={() => setCurrentPage(page)} className={currentPage === page ? 'active' : ''}>
                            {page}
                        </button>
                    ))}
                </div>
            )}

            {selectedPost && (
                <div className="modal-overlay" onClick={() => setSelectedPost(null)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <h3>Post Details</h3>
                        <pre>{JSON.stringify(selectedPost, null, 2)}</pre>
                        <button onClick={() => setSelectedPost(null)}>Close</button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ReadingPane; 