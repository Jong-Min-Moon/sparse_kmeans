from sentence_transformers import SentenceTransformer

# Load the EmbeddingGemma model from Hugging Face
model = SentenceTransformer("google/embeddinggemma-300m")

queries = [
    "How can Graph Neural Networks be used to accelerate solvers?",
    "Convergence properties of Thompson Sampling"
]

# Generate embeddings
embeddings = model.encode(queries)

for i, embedding in enumerate(embeddings):
    print(f"Query {i+1} embedded. Dimensionality: {len(embedding)}") # Will output 768
