documents = [
    "Solar production is usually highest between July and September.",
    "Solar radiation strongly affects solar production.",
    "Rain and snow can reduce solar production.",
    "Revenue depends on both production and pool price.",
    "High production does not always mean highest revenue.",
    "Seasonal patterns affect solar generation."
]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "Solar production is usually highest between July and September.",
    "Solar radiation strongly affects solar production.",
    "Rain and snow can reduce solar production.",
    "Revenue depends on both production and pool price.",
    "High production does not always mean highest revenue.",
    "Seasonal patterns affect solar generation."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)

def get_answer(query):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    best_index = np.argmax(similarities)
    return documents[best_index]
