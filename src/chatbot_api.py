"""
FastAPI Chatbot for Movie Recommendations with Poster Generation
"""
import os
import sys
from pathlib import Path
from typing import Optional
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MOVIE_DATABASE, PROMPT_TEMPLATES, GENERATED_DIR, CHATBOT_HOST, CHATBOT_PORT
from src.generate_posters import PosterGenerator

# Initialize FastAPI app
app = FastAPI(title="Movie Chatbot API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for generated posters
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")

# Initialize poster generator (lazy loading)
poster_generator = None


def get_poster_generator():
    """Lazy load poster generator"""
    global poster_generator
    if poster_generator is None:
        print("Initializing poster generator...")
        poster_generator = PosterGenerator()
    return poster_generator


# Request/Response models
class MovieRequest(BaseModel):
    query: str
    genre: Optional[str] = None


class MovieResponse(BaseModel):
    query: str
    genre: str
    recommendations: list[str]
    poster_url: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Helper functions
def detect_genre(query: str) -> str:
    """
    Simple genre detection from query text
    
    Args:
        query: User's query text
        
    Returns:
        Detected genre or 'action' as default
    """
    query_lower = query.lower()
    
    # Check for genre keywords
    for genre in MOVIE_DATABASE.keys():
        if genre in query_lower:
            return genre
    
    # Check for related keywords
    keywords = {
        "action": ["explosion", "fight", "battle", "war", "hero"],
        "sci-fi": ["space", "future", "robot", "alien", "technology", "cyberpunk"],
        "horror": ["scary", "ghost", "monster", "nightmare", "dark", "evil"],
        "comedy": ["funny", "laugh", "humor", "hilarious", "joke"],
        "drama": ["emotional", "serious", "real", "life", "story"],
        "fantasy": ["magic", "wizard", "dragon", "mystical", "enchanted"],
        "thriller": ["suspense", "mystery", "detective", "crime", "investigation"],
        "romance": ["love", "romantic", "couple", "relationship", "heart"],
    }
    
    for genre, words in keywords.items():
        if any(word in query_lower for word in words):
            return genre
    
    return "action"  # Default


def get_recommendations(genre: str, num_recommendations: int = 4) -> list[str]:
    """
    Get movie recommendations for a genre
    
    Args:
        genre: Movie genre
        num_recommendations: Number of recommendations to return
        
    Returns:
        List of movie titles
    """
    if genre in MOVIE_DATABASE:
        movies = MOVIE_DATABASE[genre]
        return movies[:num_recommendations]
    else:
        # Return random movies if genre not found
        return MOVIE_DATABASE["action"][:num_recommendations]


def generate_poster_for_query(query: str, genre: str) -> str:
    """
    Generate a movie poster based on query and genre
    
    Args:
        query: User's query
        genre: Detected genre
        
    Returns:
        Path to generated poster image
    """
    generator = get_poster_generator()
    
    # Create prompt based on genre and query
    if genre in PROMPT_TEMPLATES:
        base_prompt = PROMPT_TEMPLATES[genre]
    else:
        base_prompt = f"{genre} movie poster with cinematic style"
    
    # Incorporate user query if specific
    if len(query) > 10 and genre not in query.lower():
        prompt = f"{base_prompt}, {query}, professional movie poster"
    else:
        prompt = f"{base_prompt}, high quality, professional design"
    
    # Generate poster
    print(f"Generating poster: {prompt[:80]}...")
    image = generator.generate(prompt, guidance_scale=7.5, num_inference_steps=30)
    
    # Save with unique filename
    filename = f"poster_{uuid.uuid4().hex[:8]}.jpg"
    filepath = GENERATED_DIR / filename
    image.save(filepath, quality=95)
    
    return f"/generated/{filename}"


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "running",
        "model_loaded": poster_generator is not None
    }


@app.post("/recommend", response_model=MovieResponse)
async def recommend_movie(request: MovieRequest):
    """
    Main endpoint: Get movie recommendations and generate poster
    
    Args:
        request: MovieRequest with query and optional genre
        
    Returns:
        MovieResponse with recommendations and poster URL
    """
    try:
        # Detect genre
        if request.genre:
            genre = request.genre.lower()
        else:
            genre = detect_genre(request.query)
        
        # Get recommendations
        recommendations = get_recommendations(genre)
        
        # Generate poster
        poster_url = generate_poster_for_query(request.query, genre)
        
        # Create response message
        message = f"Based on your interest in {genre} movies, here are some recommendations!"
        
        return {
            "query": request.query,
            "genre": genre,
            "recommendations": recommendations,
            "poster_url": poster_url,
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/genres")
async def get_genres():
    """Get list of available genres"""
    return {
        "genres": list(MOVIE_DATABASE.keys())
    }


@app.post("/generate-poster")
async def generate_custom_poster(prompt: str, genre: Optional[str] = "action"):
    """
    Generate a custom poster from a direct prompt
    
    Args:
        prompt: Text prompt for poster generation
        genre: Optional genre for base style
        
    Returns:
        URL to generated poster
    """
    try:
        generator = get_poster_generator()
        
        # Enhance prompt
        full_prompt = f"{prompt}, movie poster, professional design, high quality"
        
        # Generate
        image = generator.generate(full_prompt)
        
        # Save
        filename = f"custom_poster_{uuid.uuid4().hex[:8]}.jpg"
        filepath = GENERATED_DIR / filename
        image.save(filepath, quality=95)
        
        return {
            "poster_url": f"/generated/{filename}",
            "prompt": prompt
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating poster: {str(e)}")


def main():
    """Run the FastAPI server"""
    print("=" * 60)
    print("Movie Chatbot API Server")
    print("=" * 60)
    print(f"\nStarting server at http://{CHATBOT_HOST}:{CHATBOT_PORT}")
    print(f"API docs available at http://{CHATBOT_HOST}:{CHATBOT_PORT}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        app,
        host=CHATBOT_HOST,
        port=CHATBOT_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
