import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const analyzePage = async (keyword, url, country, secondaryKeywords) => {
    try {
        const response = await api.post('/api/analyze', {
            keyword,
            url,
            country,
            secondary_keywords: secondaryKeywords,
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Analysis failed');
    }
};

export const generateContent = async (keyword, url, lsiKeywords, settings) => {
    try {
        const response = await api.post('/api/generate-content', {
            keyword,
            url,
            lsi_keywords: lsiKeywords,
            settings,
        });
        return response.data.content;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Content generation failed');
    }
};

export const scoreContent = async (html, tfidfKeywords, lsiKeywords, keywordList) => {
    try {
        const response = await api.post('/api/score', {
            html,
            tfidf_keywords: tfidfKeywords,
            lsi_keywords: lsiKeywords,
            keyword_list: keywordList,
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Scoring failed');
    }
};

export const generateHeadlines = async (keyword) => {
    try {
        const response = await api.post('/api/generate-headlines', null, {
            params: { keyword },
        });
        return response.data.headlines;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Headline generation failed');
    }
};

export const buildTopicClusters = async (keyword) => {
    try {
        const response = await api.post('/api/build-topic-clusters', null, {
            params: { keyword },
        });
        return response.data.clusters;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Topic cluster generation failed');
    }
};

export const validateOutline = async (headings) => {
    try {
        const response = await api.post('/api/validate-outline', { headings });
        return response.data.analysis;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Outline validation failed');
    }
};

export const detectFluff = async (content) => {
    try {
        const response = await api.post('/api/detect-fluff', { content });
        return response.data.findings;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Fluff detection failed');
    }
};

export const getCompressionScore = async (content) => {
    try {
        const response = await api.post('/api/compression-score', { content });
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Compression score calculation failed');
    }
}; 