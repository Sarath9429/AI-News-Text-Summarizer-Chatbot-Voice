import faiss

dimension = 768  # Gemini Pro embedding size
index = faiss.IndexFlatL2(dimension)

# Example: Store embeddings for 300 articles
embeddings = np.array([get_embedding(text) for text in extracted_text_list])  # Convert all texts to embeddings
index.add(embeddings)  # Add to FAISS index

# Save FAISS index
faiss.write_index(index, "text_embeddings.index")
