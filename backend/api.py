from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from seo import Api  # Import our existing SEO functionality

app = FastAPI(title="SEO Beast API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our SEO API
seo_api = Api()

# Request/Response Models
class AnalysisRequest(BaseModel):
    keyword: str
    url: Optional[str] = None
    country: str = "Australia"
    secondary_keywords: List[str] = []

class ContentGenerationRequest(BaseModel):
    keyword: str
    url: Optional[str] = None
    lsi_keywords: List[List[str]] = []
    settings: dict

class ScoreRequest(BaseModel):
    html: str
    tfidf_keywords: List[List[str]]
    lsi_keywords: List[List[str]]
    keyword_list: List[str]

# API Endpoints
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    try:
        result = seo_api.analyse({
            "keyword": request.keyword,
            "url": request.url,
            "country": request.country,
            "secondary_keywords": request.secondary_keywords
        })
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-content")
async def generate_content(request: ContentGenerationRequest):
    try:
        result = seo_api.generate_content({
            "keyword": request.keyword,
            "url": request.url,
            "lsi_keywords": request.lsi_keywords,
            "settings": request.settings
        })
        return {"content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/score")
async def score(request: ScoreRequest):
    try:
        score, reasons = seo_api.score(
            request.html,
            request.tfidf_keywords,
            request.lsi_keywords,
            request.keyword_list
        )
        return {"score": score, "reasons": reasons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-headlines")
async def generate_headlines(keyword: str):
    try:
        headlines = seo_api.generate_headline_suggestions(keyword)
        return {"headlines": headlines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/build-topic-clusters")
async def build_topic_clusters(keyword: str):
    try:
        clusters = seo_api.build_topic_clusters(keyword)
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validate-outline")
async def validate_outline(headings: List[str]):
    try:
        analysis = seo_api.validate_outline(headings)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-fluff")
async def detect_fluff(content: str):
    try:
        findings = seo_api.detect_fluff(content)
        return {"findings": findings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compression-score")
async def compression_score(content: str):
    try:
        score = seo_api.compression_score(content)
        return score
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 