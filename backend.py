from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json, numpy as np, faiss, os, google.generativeai as genai
from sentence_transformers import SentenceTransformer
from gtts import gTTS

# ========== CONFIG ==========
BASE_PATH = r"C:\Users\Sarath S\Desktop\Total_Files\ai-chatbot"
ARTICLES_JSON = os.path.join(BASE_PATH, "articles.json")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.npy")
TEXTS_JSON = os.path.join(BASE_PATH, "texts.json")
INDEX_FILE = os.path.join(BASE_PATH, "text_embeddings.index")
AUDIO_FILE = os.path.join(BASE_PATH, "response.mp3")

# Google Gemini API
genai.configure(api_key="AIzaSyAxO91UvcwXa5_cryR2yhcw54ALrBMnJEo")
model = genai.GenerativeModel("gemini-1.5-flash")

# Local embedding model
local_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384

# Load embeddings & FAISS index once
texts = json.load(open(TEXTS_JSON, "r", encoding="utf-8"))
embeddings = np.load(EMBEDDINGS_FILE)
index = faiss.read_index(INDEX_FILE)

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def get_embedding(text):
    return local_model.encode(text)

def retrieve_similar_texts(query):
    q_emb = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(q_emb, 3)
    return [texts[i] for i in indices[0] if i < len(texts)]

def generate_response(query, context):
    try:
        resp = model.generate_content(f"Query: {query}\n\nContext: {context}")
        return resp.text
    except:
        return "Sorry, I couldn't generate a response."

def summarize_text(text):
    try:
        resp = model.generate_content(f"Summarize this: {text}")
        return resp.text
    except:
        return "Sorry, I couldn't summarize the text."

def text_to_speech(text, filename=AUDIO_FILE):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# API routes
@app.post("/chat")
async def chat(data: dict):
    query = data.get("query", "")
    summarize = data.get("summarize", False)
    tts = data.get("tts", False)

    retrieved = retrieve_similar_texts(query)
    context = " ".join(retrieved) if retrieved else ""
    response = generate_response(query, context)

    summary_text = summarize_text(context) if summarize else None
    audio_path = text_to_speech(response) if tts else None

    return {
        "response": response,
        "summary": summary_text,
        "audio_url": "/audio" if tts else None
    }

@app.get("/audio")
async def get_audio():
    if os.path.exists(AUDIO_FILE):
        return FileResponse(AUDIO_FILE, media_type="audio/mpeg", filename="response.mp3")
    return {"error": "No audio file found."}
