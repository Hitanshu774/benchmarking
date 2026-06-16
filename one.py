import json
import numpy as np
import time
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# You will need to pip install these:
# pip install scikit-learn rank_bm25 sentence-transformers numpy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. DATA LOADING & PREPARATION ---
def load_and_chunk_corpus(filepath: str = "small_corpus.jsonl") -> List[Dict]:
    """Loads the JSONL and splits long documents into smaller chunks."""
    chunks = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                doc = json.loads(line)
                # Simple sentence/period splitting for chunking
                text_splits = doc.get("text", "").split(". ")
                
                # Group every 3 sentences into a chunk
                for i in range(0, len(text_splits), 3):
                    chunk_text = ". ".join(text_splits[i:i+3]).strip()
                    if len(chunk_text) > 20: # Ignore tiny fragments
                        chunks.append({
                            "chunk_id": f"{doc['_id']}_chunk_{i}",
                            "text": chunk_text
                        })
        print(f"Loaded {len(chunks)} chunks from {filepath}")
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using fallback mock data.")
        # Fallback data based on your uploaded snippet so the script never crashes
        chunks = [
            {"chunk_id": "c0_1", "text": "As part of an effort to streamline the nominations process, a standing order of the Senate, S.Res. 116, created a new designation of certain nominations as privileged."},
            {"chunk_id": "c0_2", "text": "In total, there are 285 positions to which nominations are privileged, the majority of which are part-time appointments to oversight boards."},
            {"chunk_id": "c0_3", "text": "Nearly 25 percent of the total number of persons without disabilities that were hired at SSA stayed for less than 1 year of service."},
        ]
    return chunks

# --- 2. THE GOLDEN EVALUATION SET (DYNAMIC MAPPING) ---
# Instead of hardcoding chunk_ids that might change based on parsing,
# we define the exact substring that MUST be present in the expected chunk.
RAW_EVAL_DATASET = [
    {
        "query": "Which standing order created the privileged nominations designation?", 
        "expected_substring": "S.Res. 116" # Keyword heavy query
    },
    {
        "query": "How many privileged positions exist?", 
        "expected_substring": "285 positions to which nominations are privileged" # Semantic/Factoid query
    },
    {
        "query": "What is the retention issue at the Social Security Administration for non-disabled hires?", 
        "expected_substring": "25 percent of the total number of persons without disabilities" # Complex query requiring domain mapping
    },
    # --- NEW QUERIES WHERE HYBRID SHINES ---
    {
        "query": "What fraction of non-handicapped employees left the agency within 12 months?", 
        "expected_substring": "25 percent of the total number of persons without disabilities" # Vocabulary Mismatch
    },
    {
        "query": "Are most of the expedited senate selections for full-time roles?", 
        "expected_substring": "285 positions to which nominations are privileged" # Semantic Synonym
    },
    {
        "query": "Which rule was introduced to make confirming presidential appointees faster?",
        "expected_substring": "S.Res. 116" # Conceptual Abstraction
    }
]

# --- 3. BASELINE: HASHING VECTORIZER ---
class HashingVectorizerRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        # Initialize the stateless vectorizer
        self.vectorizer = HashingVectorizer(n_features=2**12, stop_words='english')
        # Pre-compute document matrix
        self.doc_matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])
        
    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_vec = self.vectorizer.transform([query])
        # Compute cosine similarity between query and all documents
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()
        
        # Get top K indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[i]["chunk_id"] for i in top_indices]

# --- 4. CHALLENGER: HYBRID RETRIEVER (BM25 + Dense + Reranker) ---
class HybridRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        
        print("Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.embedding_model.encode(self.texts, convert_to_tensor=True)
        
        print("Loading sparse index (BM25)...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Loading Reranker (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search(self, query: str, top_k: int = 3) -> List[str]:
        # 1. Sparse Retrieval (Top 10)
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[-10:]
        
        # 2. Dense Retrieval (Top 10)
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        # We use dot product for cosine similarity with normalized embeddings
        from sentence_transformers.util import cos_sim
        dense_scores = cos_sim(query_embedding, self.doc_embeddings)[0]
        top_dense_idx = np.argsort(dense_scores.cpu().numpy())[-10:]
        
        # 3. Combine unique candidates
        unique_candidates_idx = list(set(top_bm25_idx).union(set(top_dense_idx)))
        candidate_texts = [self.texts[i] for i in unique_candidates_idx]
        
        # 4. Rerank
        cross_inp = [[query, text] for text in candidate_texts]
        cross_scores = self.reranker.predict(cross_inp)
        
        # Sort candidates by reranker score
        best_indices = np.argsort(cross_scores)[::-1][:top_k]
        
        # Map back to original chunk IDs
        final_chunk_ids = [self.chunks[unique_candidates_idx[i]]["chunk_id"] for i in best_indices]
        return final_chunk_ids

# --- 5. EVALUATION HARNESS ---
def calculate_metrics(retrieved_ids: List[str], expected_id: str) -> Tuple[int, float]:
    hit = 1 if expected_id in retrieved_ids else 0
    mrr = 0.0
    if expected_id in retrieved_ids:
        rank = retrieved_ids.index(expected_id) + 1
        mrr = 1.0 / rank
    return hit, mrr

def run_benchmark():
    chunks = load_and_chunk_corpus()
    
    print("\n--- Mapping Golden Dataset to Dynamic Chunk IDs ---")
    eval_dataset = []
    for item in RAW_EVAL_DATASET:
        mapped_id = None
        for c in chunks:
            if item["expected_substring"].lower() in c["text"].lower():
                mapped_id = c["chunk_id"]
                break
        
        if mapped_id:
            eval_dataset.append({
                "query": item["query"],
                "expected_chunk_id": mapped_id
            })
        else:
            print(f"⚠️ Warning: Could not find ground truth in corpus for query: '{item['query']}'")

    if not eval_dataset:
        print("Error: No queries could be mapped to the current corpus. Exiting.")
        return

    print("\n--- Initializing Retrievers ---")
    hashing_retriever = HashingVectorizerRetriever(chunks)
    hybrid_retriever = HybridRetriever(chunks)
    
    results = {"Hashing": {"hits": 0, "mrr": 0.0, "time": 0.0}, "Hybrid": {"hits": 0, "mrr": 0.0, "time": 0.0}}
    total_queries = len(eval_dataset)
    
    print("\n--- Running Evaluation ---")
    for i, item in enumerate(eval_dataset):
        query = item["query"]
        expected = item["expected_chunk_id"]
        print(f"\nQ{i+1}: '{query}'")
        print(f"Target Chunk ID: {expected}")
        
        # Test Hashing
        start_time = time.time()
        hash_results = hashing_retriever.search(query, top_k=3)
        h_time = time.time() - start_time
        
        h_hit, h_mrr = calculate_metrics(hash_results, expected)
        results["Hashing"]["hits"] += h_hit
        results["Hashing"]["mrr"] += h_mrr
        results["Hashing"]["time"] += h_time
        print(f" [Hashing] Retrieved: {hash_results} | Hit: {h_hit} | MRR: {h_mrr:.2f} | Time: {h_time:.4f}s")
        
        # Test Hybrid
        start_time = time.time()
        hybrid_results = hybrid_retriever.search(query, top_k=3)
        hy_time = time.time() - start_time
        
        hy_hit, hy_mrr = calculate_metrics(hybrid_results, expected)
        results["Hybrid"]["hits"] += hy_hit
        results["Hybrid"]["mrr"] += hy_mrr
        results["Hybrid"]["time"] += hy_time
        print(f" [Hybrid]  Retrieved: {hybrid_results} | Hit: {hy_hit} | MRR: {hy_mrr:.2f} | Time: {hy_time:.4f}s")

    print("\n" + "="*55)
    print("🏆 BENCHMARK RESULTS 🏆")
    print("="*55)
    print(f"Total Queries: {total_queries}")
    for name, metrics in results.items():
        hit_rate = (metrics['hits'] / total_queries) * 100
        avg_mrr = metrics['mrr'] / total_queries
        avg_time = metrics['time'] / total_queries
        print(f"{name:10} | Hit Rate @ 3: {hit_rate:5.1f}% | Avg MRR: {avg_mrr:.3f} | Avg Latency: {avg_time:.4f}s")
    print("="*55)

if __name__ == "__main__":
    run_benchmark()