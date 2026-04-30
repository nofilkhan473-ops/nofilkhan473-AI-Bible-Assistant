import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr

# ==========================================
# CONFIGURATION 
# ==========================================
# Note: Hugging Face automatically pulls your GROQ_API_KEY from Settings > Secrets!

EMBEDDINGS_DIR = "embeddings"
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "bible_index.faiss")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.json")

print("Loading Embedding Model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_and_preprocess_dataset():
    """Downloads the Complete King James Bible from a verified working GitHub URL."""
    print("Downloading the Complete KJV Bible Dataset...")
    
    url = "https://raw.githubusercontent.com/Jcharis/Bible-NLP-and-ML-using-Python/master/t_kjv.csv"
    
    df = pd.read_csv(url)
    
    book_mapping = {
        1: "Genesis", 2: "Exodus", 3: "Leviticus", 4: "Numbers", 5: "Deuteronomy",
        6: "Joshua", 7: "Judges", 8: "Ruth", 9: "1 Samuel", 10: "2 Samuel",
        11: "1 Kings", 12: "2 Kings", 13: "1 Chronicles", 14: "2 Chronicles", 15: "Ezra",
        16: "Nehemiah", 17: "Esther", 18: "Job", 19: "Psalms", 20: "Proverbs",
        21: "Ecclesiastes", 22: "Song of Solomon", 23: "Isaiah", 24: "Jeremiah", 25: "Lamentations",
        26: "Ezekiel", 27: "Daniel", 28: "Hosea", 29: "Joel", 30: "Amos",
        31: "Obadiah", 32: "Jonah", 33: "Micah", 34: "Nahum", 35: "Habakkuk",
        36: "Zephaniah", 37: "Haggai", 38: "Zechariah", 39: "Malachi", 40: "Matthew",
        41: "Mark", 42: "Luke", 43: "John", 44: "Acts", 45: "Romans",
        46: "1 Corinthians", 47: "2 Corinthians", 48: "Galatians", 49: "Ephesians", 50: "Philippians",
        51: "Colossians", 52: "1 Thessalonians", 53: "2 Thessalonians", 54: "1 Timothy", 55: "2 Timothy",
        56: "Titus", 57: "Philemon", 58: "Hebrews", 59: "James", 60: "1 Peter",
        61: "2 Peter", 62: "1 John", 63: "2 John", 64: "3 John", 65: "Jude", 66: "Revelation"
    }
    
    metadata_list = []
    for _, row in df.iterrows():
        metadata_list.append({
            "text": str(row['t']).strip(),
            "book": book_mapping.get(row['b'], f"Book {row['b']}"),
            "chapter": str(row['c']),
            "verse": str(row['v'])
        })
        
    print(f"Successfully loaded {len(metadata_list)} complete Bible verses.")
    return metadata_list

def build_and_save_embeddings(metadata_list):
    print("Creating embeddings directory...")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    docs = [item["text"] for item in metadata_list]
    print("Generating embeddings (this will take a few minutes)...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    
    print("Building FAISS vector index...")
    vector_index = faiss.IndexFlatL2(embeddings.shape[1])
    vector_index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(vector_index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    print("Successfully built and saved all embeddings!")

def load_saved_embeddings():
    print("Loading saved FAISS index and metadata...")
    vector_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    return vector_index, metadata_list

def retrieve_context(query, vector_index, metadata_list, top_k=5):
    query_vector = embedding_model.encode([query]).astype('float32')
    distances, indices = vector_index.search(query_vector, top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(metadata_list):
            results.append(metadata_list[idx])
    return results

def generate_response(query, retrieved_data):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    context_str = ""
    for item in retrieved_data:
        context_str += f"[{item['book']} {item['chapter']}:{item['verse']}]\nText: {item['text']}\n\n"
        
    system_prompt = """You are a strict data-extraction assistant. You are NOT a general AI. 
    You have ONE job: Answer the user's query ONLY by quoting the provided 'Context (Retrieved Verses)'.
    
    CRITICAL RULES:
    1. ZERO OUTSIDE KNOWLEDGE: You are forbidden from using your pre-trained knowledge. 
    2. MISSING INFORMATION: Read the context carefully. If the verses have nothing to do with the user's query, reply EXACTLY with: "I'm sorry, but the retrieved verses from the dataset do not contain information about this." 
    3. QUOTING & CITATIONS: If you find relevant verses in the context, you must output the exact text of the verse. You MUST append the exact reference after it like this: (Book Name Chapter:Verse).
    """
    
    user_prompt = f"Context (Retrieved Verses):\n{context_str}\n\nUser Query: {query}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0, 
        max_tokens=1024
    )
    return response.choices[0].message.content

def chat_wrapper(user_message, history):
    try:
        retrieved_verses = retrieve_context(user_message, global_index, global_metadata)
        return generate_response(user_message, retrieved_verses)
    except Exception as e:
        return f"An error occurred: {str(e)}"

print("--- Starting Bible RAG Application ---")

# Database fail-safe check
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
    global_index, global_metadata = load_saved_embeddings()
else:
    raw_metadata = load_and_preprocess_dataset()
    build_and_save_embeddings(raw_metadata)
    global_index, global_metadata = load_saved_embeddings()

# Launch Gradio interface
demo = gr.ChatInterface(
    fn=chat_wrapper,
    title="📖 AI Bible Study Assistant",
    description="**⚠️ Disclaimer:** *This application is for testing purposes only. AI can sometimes generate irrelevant or inaccurate answers. We hold deep respect for all religions and beliefs.* <br><br> Ask questions in English. The app retrieves verses from the King James Bible dataset.",
    examples=["What does the Bible say about divorce?", "Who was Moses?"]
)

demo.launch()
