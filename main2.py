import json
import numpy as np
import google.generativeai as genai
import faiss
from gtts import gTTS
import os
import pygame
from sentence_transformers import SentenceTransformer  # Local model

# ========== CONFIG ==========
BASE_PATH = r"C:\Users\Sarath S\Desktop\Total_Files\ai-chatbot"
TEXT_FILE = os.path.join(BASE_PATH, "extracted_text_paddle.txt")
ARTICLES_JSON = os.path.join(BASE_PATH, "articles.json")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.npy")
TEXTS_JSON = os.path.join(BASE_PATH, "texts.json")
INDEX_FILE = os.path.join(BASE_PATH, "text_embeddings.index")

# Google Gemini API
genai.configure(api_key="AIzaSyAxO91UvcwXa5_cryR2yhcw54ALrBMnJEo")  
model = genai.GenerativeModel("gemini-1.5-flash")

# Local embedding model
local_model = SentenceTransformer("all-MiniLM-L6-v2")  


# 🔹 Step 1: Convert Extracted Text into JSON
def create_articles_json():
    try:
        with open(TEXT_FILE, "r", encoding="utf-8") as file:
            data = file.read().split("[Extracted from")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {TEXT_FILE}")
        exit()

    articles = []
    for entry in data[1:]:
        lines = entry.split("\n", 1)
        image_name = lines[0].strip(" ]") if len(lines) > 1 else "unknown"
        content = lines[1].strip() if len(lines) > 1 else ""
        if content.strip():
            articles.append({"image": image_name, "text": content})

    with open(ARTICLES_JSON, "w", encoding="utf-8") as json_file:
        json.dump(articles, json_file, indent=4, ensure_ascii=False)

    print(f"✅ Extracted text saved in '{ARTICLES_JSON}'")


if not os.path.exists(ARTICLES_JSON):
    create_articles_json()


# 🔹 Step 2: Generate Embeddings
def get_embedding(text):
    """Generate embeddings using local model (fast)."""
    return local_model.encode(text)


def generate_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        print("⚡ Using pre-generated embeddings")
        return

    with open(ARTICLES_JSON, "r", encoding="utf-8") as json_file:
        articles = json.load(json_file)

    texts = [article["text"] for article in articles if article["text"].strip()]
    embeddings = np.array([get_embedding(text) for text in texts])

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(TEXTS_JSON, "w", encoding="utf-8") as text_file:
        json.dump(texts, text_file, indent=4)

    print("✅ Embeddings saved successfully!")


if not os.path.exists(EMBEDDINGS_FILE):
    generate_embeddings()


# 🔹 Step 3: Load / Create FAISS Index
dimension = 384  # all-MiniLM-L6-v2 output size


def load_faiss_index():
    if os.path.exists(INDEX_FILE):
        print("⚡ Loading FAISS index...")
        return faiss.read_index(INDEX_FILE)
    else:
        print("🚀 Creating new FAISS index...")
        return faiss.IndexFlatL2(dimension)


index = load_faiss_index()


def store_embeddings():
    embeddings = np.load(EMBEDDINGS_FILE)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("✅ FAISS index saved!")


if not os.path.exists(INDEX_FILE):
    store_embeddings()


# 🔹 Step 4: Retrieve Relevant Texts
def retrieve_similar_texts(query):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, 3)

    with open(TEXTS_JSON, "r", encoding="utf-8") as text_file:
        texts = json.load(text_file)

    retrieved_texts = [texts[i] for i in indices[0] if i < len(texts)]
    return retrieved_texts


# 🔹 Step 5: Generate Response with Gemini
def generate_response(query, context):
    try:
        response = model.generate_content(f"Query: {query}\n\nContext: {context}")
        return response.text
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Sorry, I couldn't generate a response."


# 🔹 Step 6: Summarization
def summarize_text(text):
    try:
        response = model.generate_content(f"Summarize this: {text}")
        return response.text
    except Exception as e:
        print(f"❌ Error summarizing text: {e}")
        return "Sorry, I couldn't summarize the text."


# 🔹 Step 7: Text-to-Speech
def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# 🔹 Step 8: Run the Chatbot
if __name__ == "__main__":
    user_query = input("Ask a question: ")
    retrieved_texts = retrieve_similar_texts(user_query)

    if retrieved_texts:
        context = " ".join(retrieved_texts)
        response = generate_response(user_query, context)
        print("\n💬 AI Response:", response)

        summarize = input("\nDo you want a summary? (yes/no): ").strip().lower()
        if summarize == "yes":
            summary = summarize_text(context)
            print("\n📌 Summary:", summary)

        tts_option = input("\nDo you want text-to-speech? (yes/no): ").strip().lower()
        if tts_option == "yes":
            text_to_speech(response)
    else:
        print("\n❌ No relevant text found. Try another question.")