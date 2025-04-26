import React, { useState, useEffect } from 'react';
import { 
    analyzePage, 
    generateContent, 
    scoreContent, 
    generateHeadlines,
    buildTopicClusters,
    validateOutline,
    detectFluff,
    getCompressionScore
} from '../services/api';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css';

const SEOAnalyzer = () => {
    const [keyword, setKeyword] = useState('');
    const [url, setUrl] = useState('');
    const [country, setCountry] = useState('de');
    const [secondaryKeywords, setSecondaryKeywords] = useState([]);
    const [content, setContent] = useState('');
    const [score, setScore] = useState(0);
    const [scoreReasons, setScoreReasons] = useState([]);
    const [headlines, setHeadlines] = useState([]);
    const [topicClusters, setTopicClusters] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleAnalyze = async () => {
        try {
            setIsLoading(true);
            setError(null);
            
            const analysis = await analyzePage(keyword, url, country, secondaryKeywords);
            setContent(analysis.content);
            setScore(analysis.score);
            setScoreReasons(analysis.reasons);
            
            const generatedHeadlines = await generateHeadlines(keyword);
            setHeadlines(generatedHeadlines);
            
            const clusters = await buildTopicClusters(keyword);
            setTopicClusters(clusters);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleContentChange = async (newContent) => {
        setContent(newContent);
        try {
            const scoreResult = await scoreContent(
                newContent,
                secondaryKeywords,
                [],
                [keyword, ...secondaryKeywords]
            );
            setScore(scoreResult.score);
            setScoreReasons(scoreResult.reasons);
        } catch (err) {
            setError(err.message);
        }
    };

    const handleGenerateContent = async () => {
        try {
            setIsLoading(true);
            setError(null);
            
            const generatedContent = await generateContent(keyword, url, [], {
                wordCount: 1500,
                headings: 5,
                paragraphs: 10,
                images: 3
            });
            setContent(generatedContent);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleValidateOutline = async () => {
        try {
            const headings = content.match(/<h[1-6][^>]*>.*?<\/h[1-6]>/g) || [];
            const analysis = await validateOutline(headings);
            setScoreReasons(prev => [...prev, ...analysis]);
        } catch (err) {
            setError(err.message);
        }
    };

    const handleDetectFluff = async () => {
        try {
            const findings = await detectFluff(content);
            setScoreReasons(prev => [...prev, ...findings]);
        } catch (err) {
            setError(err.message);
        }
    };

    const handleGetCompressionScore = async () => {
        try {
            const compressionScore = await getCompressionScore(content);
            setScoreReasons(prev => [...prev, compressionScore]);
        } catch (err) {
            setError(err.message);
        }
    };

    return (
        <div className="seo-analyzer">
            <div className="input-section">
                <input
                    type="text"
                    value={keyword}
                    onChange={(e) => setKeyword(e.target.value)}
                    placeholder="Enter keyword"
                />
                <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="Enter URL"
                />
                <select value={country} onChange={(e) => setCountry(e.target.value)}>
                    <option value="de">Germany</option>
                    <option value="us">United States</option>
                    <option value="uk">United Kingdom</option>
                </select>
                <button onClick={handleAnalyze} disabled={isLoading}>
                    Analyze
                </button>
            </div>

            {error && <div className="error">{error}</div>}

            <div className="score-section">
                <h2>SEO Score: {score}/100</h2>
                <ul>
                    {scoreReasons.map((reason, index) => (
                        <li key={index}>{reason}</li>
                    ))}
                </ul>
            </div>

            <div className="content-section">
                <div className="editor-controls">
                    <button onClick={handleGenerateContent} disabled={isLoading}>
                        Generate Content
                    </button>
                    <button onClick={handleValidateOutline}>Validate Outline</button>
                    <button onClick={handleDetectFluff}>Detect Fluff</button>
                    <button onClick={handleGetCompressionScore}>Get Compression Score</button>
                </div>
                <ReactQuill
                    value={content}
                    onChange={handleContentChange}
                    modules={{
                        toolbar: [
                            [{ 'header': [1, 2, 3, false] }],
                            ['bold', 'italic', 'underline', 'strike'],
                            ['link', 'image'],
                            ['clean']
                        ]
                    }}
                />
            </div>

            <div className="headlines-section">
                <h3>Suggested Headlines</h3>
                <ul>
                    {headlines.map((headline, index) => (
                        <li key={index}>{headline}</li>
                    ))}
                </ul>
            </div>

            <div className="topic-clusters-section">
                <h3>Topic Clusters</h3>
                <ul>
                    {topicClusters.map((cluster, index) => (
                        <li key={index}>
                            <h4>{cluster.topic}</h4>
                            <ul>
                                {cluster.subtopics.map((subtopic, subIndex) => (
                                    <li key={subIndex}>{subtopic}</li>
                                ))}
                            </ul>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default SEOAnalyzer; 