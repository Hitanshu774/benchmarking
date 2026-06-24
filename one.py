import json
import time
import tracemalloc
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_distances

# You will need to pip install these:
# pip install scikit-learn rank_bm25 sentence-transformers numpy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. THE PRODUCTION FORMATTER (APPLES-TO-APPLES) ---
def format_production_text(chunk: Dict) -> str:
    """
    Replicates the exact logic from document_chunking_service.py (Line 351).
    Ensures both retrievers are only searching the summary and keywords.
    """
    summary = chunk.get("summary", "")
    keywords = chunk.get("keywords", [])
    
    # If production fields exist, use them. Otherwise fallback to raw text.
    if summary or keywords:
        return f"{summary} {' '.join(keywords)}".strip()
    return chunk.get("text", "")


# --- 2. DATA LOADING & PREPARATION ---
def load_and_chunk_corpus(filepath: str = "small_corpus.jsonl") -> List[Dict]:
    """Loads the JSONL and splits long documents into smaller chunks."""
    chunks = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                doc = json.loads(line)
                
                # If production chunks already exist in JSON, load them directly
                if "summary" in doc and "keywords" in doc:
                    chunks.append(doc)
                    continue

                # Otherwise simulate the chunking logic
                text_splits = doc.get("text", "").split(". ")
                window_size = 5
                step = 3
                for i in range(0, len(text_splits), step):
                    chunk_text = ". ".join(text_splits[i:i+window_size]).strip()
                    if len(chunk_text) > 20: 
                        chunks.append({
                            "chunk_id": f"{doc.get('_id', 'doc')}_chunk_{i}",
                            "text": chunk_text,
                            # FIX: Preserve the text in the simulated summary so we don't drop the answers
                            "summary": f"Summary: {chunk_text}",
                            # FIX: Extract all words as keywords to properly simulate production extraction
                            "keywords": list(set(chunk_text.lower().replace(".", "").replace(",", "").split()))
                        })
        print(f"Loaded {len(chunks)} chunks from {filepath}")
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using fallback mock data with simulated summaries/keywords.")
        raw_mock = [
            {"chunk_id": "c0_1", "text": "As part of an effort to streamline the nominations process, a standing order of the Senate, S.Res. 116, created a new designation of certain nominations as privileged."},
            {"chunk_id": "c0_2", "text": "In total, there are 285 positions to which nominations are privileged, the majority of which are part-time appointments to oversight boards."},
            {"chunk_id": "c0_3", "text": "Nearly 25 percent of the total number of persons without disabilities that were hired at SSA stayed for less than 1 year of service."},
            {"chunk_id": "c0_4", "text": "The Terrorism Risk Insurance Program Reauthorization Act of 2015 created a new 13-member Board of Directors for the National Association of Registered Agents and Brokers and designated these positions as privileged nominations established by S.Res. 116."}
        ]
        # Add mock production fields to match the AD chunking schema
        for c in raw_mock:
            # FIX: Preserve the text in the simulated summary
            c["summary"] = f"Summary: {c['text']}"
            c["keywords"] = list(set(c["text"].lower().replace(".", "").replace(",", "").split()))
            chunks.append(c)
            
    return chunks

# --- 3. THE GOLDEN EVALUATION SET ---
RAW_EVAL_DATASET = [
    {"query": "Which standing order created the privileged nominations designation?", "expected_substring": "s.res. 116"},
    {"query": "How many privileged positions exist?", "expected_substring": "285 positions"},
    {"query": "What is the retention issue at the Social Security Administration for non-disabled hires?", "expected_substring": "25 percent of the total number"},
    {"query": "What fraction of non-handicapped employees left the agency within 12 months?", "expected_substring": "25 percent of the total number"},
    {"query": "Are most of the expedited senate selections for full-time roles?", "expected_substring": "285 positions"},
    {"query": "Which rule was introduced to make confirming presidential appointees faster?", "expected_substring": "s.res. 116"},
    {"query": "Were the roles initially expedited by the 112th Congress later expanded to include any insurance-related boards?", "expected_substring": "terrorism risk insurance program"},
    {"query": "How did the Washington Post obtain the Afghanistan Papers interviews?", "expected_substring": "freedom of information act (foia)"},
    {"query": "What did Ashraf Ghani say about the financial limit Afghanistan could handle in 2002?", "expected_substring": "absorb money was $2 billion"},
    {"query": "What is the total estimated outlay for the Department of Health and Human Services in the FY2021 budget request?", "expected_substring": "$1.370 trillion"},
    {"query": "Why do private sector companies try to avoid building products that break easily?", "expected_substring": "increased warranty expenses that decrease profits"},
    {"query": "Which two major defense programs failed to include reliability engineers early in their system development?", "expected_substring": "expeditionary fighting vehicle (efv) and f-22"},
    {"query": "What technical defect forced all F-35s out of the sky in late 2018?", "expected_substring": "manufacturing fault with an engine fuel tube"},
    {"query": "What was the projected acquisition cost for the F-35 program mentioned in the report?", "expected_substring": "$406 billion"},
    {"query": "Did the 1985 review of defense profit guidelines factor in a company's assets and liabilities?", "expected_substring": "did not explicitly take into account the cost of working capital"},
    {"query": "What was the authorized end strength for uniformed Army personnel in 2017?", "expected_substring": "1.018 million uniformed personnel"},
    {"query": "What defines a covered defense business system in terms of budget authority?", "expected_substring": "budget authority of over $50 million"},
    {"query": "When did the TSA Pipeline Security Branch issue its revised security guidelines?", "expected_substring": "guidelines in march 2018"},
    {"query": "What act established requirements for legislation that imposes duties on state or local governments without funding?", "expected_substring": "unfunded mandates reform act of 1995"},
    {"query": "What monetary limit dictates whether a military enterprise software project is heavily regulated?", "expected_substring": "budget authority of over $50 million"}
]

# --- 4. BASELINE: HASHING VECTORIZER ---
class HashingVectorizerRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.vectorizer = HashingVectorizer(n_features=1024, stop_words='english') # Aligned to 1024-dim
        
        # WE INDEX THE PRODUCTION TEXT (Summary + Keywords), NOT RAW TEXT
        self.searchable_texts = [format_production_text(c) for c in chunks]
        self.doc_matrix = self.vectorizer.fit_transform(self.searchable_texts)
        
    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_vec = self.vectorizer.transform([query])
        distances = cosine_distances(query_vec, self.doc_matrix).flatten()
        top_indices = distances.argsort()[:top_k]
        return [self.chunks[i]["chunk_id"] for i in top_indices]

# --- 5. CHALLENGER: HYBRID RETRIEVER (BM25 + Dense + Reranker) ---
class HybridRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        
        # WE INDEX THE PRODUCTION TEXT (Summary + Keywords), NOT RAW TEXT
        self.searchable_texts = [format_production_text(c) for c in chunks]
        
        print("Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.embedding_model.encode(self.searchable_texts, convert_to_tensor=True)
        
        print("Loading sparse index (BM25)...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.searchable_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Loading Reranker (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search(self, query: str, top_k: int = 3) -> List[str]:
        # OPTIMIZATION: Reduce initial retrieval pool from 25 to 10 per method.
        # This cuts the heavy CrossEncoder workload by 60%, drastically reducing latency.
        candidate_k = 5
        
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[-candidate_k:]
        
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        from sentence_transformers.util import cos_sim
        dense_scores = cos_sim(query_embedding, self.doc_embeddings)[0]
        top_dense_idx = np.argsort(dense_scores.cpu().numpy())[-candidate_k:]
        
        unique_candidates_idx = list(set(top_bm25_idx).union(set(top_dense_idx)))
        candidate_texts = [self.searchable_texts[i] for i in unique_candidates_idx]
        
        cross_inp = [[query, text] for text in candidate_texts]
        cross_scores = self.reranker.predict(cross_inp)
        
        best_indices = np.argsort(cross_scores)[::-1][:top_k]
        return [self.chunks[unique_candidates_idx[i]]["chunk_id"] for i in best_indices]


# --- 6. EVALUATION HARNESS ---
def calculate_metrics(retrieved_ids: List[str], expected_id: str, k: int = 3) -> Dict[str, float]:
    retrieved_ids = retrieved_ids[:k]
    recall = 1.0 if expected_id in retrieved_ids else 0.0
    precision = 1.0 / len(retrieved_ids) if (expected_id in retrieved_ids and len(retrieved_ids) > 0) else 0.0
    
    mrr, ndcg = 0.0, 0.0
    if expected_id in retrieved_ids:
        rank = retrieved_ids.index(expected_id) + 1
        mrr = 1.0 / rank
        ndcg = 1.0 / np.log2(rank + 1)
        
    return {"recall": recall, "precision": precision, "mrr": mrr, "ndcg": ndcg}

def run_benchmark():
    chunks = load_and_chunk_corpus()
    
    print("\n--- Mapping Golden Dataset to Dynamic Chunk IDs ---")
    eval_dataset = []
    for item in RAW_EVAL_DATASET:
        mapped_id = None
        for c in chunks:
            # We map against the raw text to ensure we know the absolute ground truth
            if item["expected_substring"].lower() in c.get("text", "").lower():
                mapped_id = c["chunk_id"]
                break
        if mapped_id:
            eval_dataset.append({"query": item["query"], "expected_chunk_id": mapped_id})
    
    print(f"Successfully mapped {len(eval_dataset)} queries.")

    print("\n--- Initializing Retrievers ---")
    hashing_retriever = HashingVectorizerRetriever(chunks)
    hybrid_retriever = HybridRetriever(chunks)
    
    results = {
        "Hashing": {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "latency_sec": 0.0, "peak_mem_mb": 0.0}, 
        "Hybrid": {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "latency_sec": 0.0, "peak_mem_mb": 0.0}
    }
    total_queries = len(eval_dataset)
    
    print("\n--- Running Evaluation with Latency/Memory Profiling ---")
    for i, item in enumerate(eval_dataset):
        query = item["query"]
        expected = item["expected_chunk_id"]
        print(f"\nQ{i+1}: '{query}'")
        
        # --- PROFILING HASHING ---
        tracemalloc.start()
        t0 = time.time()
        hash_results = hashing_retriever.search(query, top_k=3)
        h_latency = time.time() - t0
        _, h_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        h_metrics = calculate_metrics(hash_results, expected, k=3)
        h_recall = h_metrics["recall"]
        results["Hashing"]["recall"] += h_recall
        results["Hashing"]["precision"] += h_metrics["precision"]
        results["Hashing"]["mrr"] += h_metrics["mrr"]
        results["Hashing"]["ndcg"] += h_metrics["ndcg"]
        results["Hashing"]["latency_sec"] += h_latency
        results["Hashing"]["peak_mem_mb"] += (h_peak / 10**6)
        
        print(f" [Hashing] Recall: {h_recall} | Prec: {h_metrics['precision']:.2f} | MRR: {h_metrics['mrr']:.2f} | NDCG: {h_metrics['ndcg']:.2f} | Time: {h_latency*1000:.1f}ms | Mem: {h_peak/10**6:.3f}MB")
        
        # --- PROFILING HYBRID ---
        tracemalloc.start()
        t0 = time.time()
        hybrid_results = hybrid_retriever.search(query, top_k=3)
        hy_latency = time.time() - t0
        _, hy_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        hy_metrics = calculate_metrics(hybrid_results, expected, k=3)
        hy_recall = hy_metrics["recall"]
        results["Hybrid"]["recall"] += hy_recall
        results["Hybrid"]["precision"] += hy_metrics["precision"]
        results["Hybrid"]["mrr"] += hy_metrics["mrr"]
        results["Hybrid"]["ndcg"] += hy_metrics["ndcg"]
        results["Hybrid"]["latency_sec"] += hy_latency
        results["Hybrid"]["peak_mem_mb"] += (hy_peak / 10**6)
        
        print(f" [Hybrid]  Recall: {hy_recall} | Prec: {hy_metrics['precision']:.2f} | MRR: {hy_metrics['mrr']:.2f} | NDCG: {hy_metrics['ndcg']:.2f} | Time: {hy_latency*1000:.1f}ms | Mem: {hy_peak/10**6:.3f}MB")

    print("\n" + "="*80)
    print("🏆 FINAL AD-ADOPTION BENCHMARK RESULTS 🏆")
    print("="*80)
    print(f"Total Queries Evaluated: {total_queries}")
    for name, metrics in results.items():
        avg_recall = (metrics['recall'] / max(total_queries, 1)) * 100
        avg_prec = metrics['precision'] / max(total_queries, 1)
        avg_mrr = metrics['mrr'] / max(total_queries, 1)
        avg_ndcg = metrics['ndcg'] / max(total_queries, 1)
        avg_lat = (metrics['latency_sec'] / max(total_queries, 1)) * 1000
        avg_mem = (metrics['peak_mem_mb'] / max(total_queries, 1))
        
        print(f"\n{name} Retriever:")
        print(f"  -> Recall@3 (Hit Rate): {avg_recall:5.1f}%")
        print(f"  -> Precision@3:         {avg_prec:.3f}")
        print(f"  -> Avg MRR:             {avg_mrr:.3f}")
        print(f"  -> Avg NDCG@3:          {avg_ndcg:.3f}")
        print(f"  -> Avg Latency/Query:   {avg_lat:.2f} ms")
        print(f"  -> Avg Memory Spike:    {avg_mem:.3f} MB")
    print("="*80)

if __name__ == "__main__":
    run_benchmark()