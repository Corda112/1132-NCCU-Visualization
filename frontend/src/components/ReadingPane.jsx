import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ReadingPane.css';

const ReadingPane = ({ filter }) => {
    const [articles, setArticles] = useState([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(0);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchArticles = async () => {
            setLoading(true);
            try {
                const response = await axios.get('http://localhost:3001/api/articles', {
                    params: { ...filter, page }
                });
                setArticles(response.data.articles);
                setTotalPages(response.data.totalPages);
            } catch (error) {
                console.error('Error fetching articles:', error);
            }
            setLoading(false);
        };

        // Fetch articles if a filter is set
        if (filter && (filter.term || filter.date)) {
            fetchArticles();
        } else {
            // Optionally, clear articles or show a message if no filter is active
            setArticles([]);
            setTotalPages(0);
        }

    }, [filter, page]);

    // Reset page to 1 when filter changes
    useEffect(() => {
        setPage(1);
    }, [filter]);

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= totalPages) {
            setPage(newPage);
        }
    };

    if (loading) {
        return <div className="loading-pane">Loading articles...</div>;
    }

    if (articles.length === 0) {
        return <div className="info-pane">Click on a data point in the 'Term' or 'N-gram' charts to see related articles here.</div>;
    }

    return (
        <div className="reading-pane">
            <div className="article-list">
                {articles.map(article => (
                    <div key={article.id} className="article-item">
                        <p className="article-text">{article.cleaned_text}</p>
                        <div className="article-meta">
                            <span>{new Date(article.createdAt).toLocaleString()}</span>
                            <span className={`sentiment-tag ${article.sentiment}`}>{article.sentiment}</span>
                        </div>
                    </div>
                ))}
            </div>
            <div className="pagination">
                <button onClick={() => handlePageChange(page - 1)} disabled={page <= 1}>
                    Previous
                </button>
                <span>Page {page} of {totalPages}</span>
                <button onClick={() => handlePageChange(page + 1)} disabled={page >= totalPages}>
                    Next
                </button>
            </div>
        </div>
    );
};

export default ReadingPane; 