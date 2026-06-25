"""
Document Chunking Service for Smart Data Access

This service intelligently segments uploaded documents into meaningful chunks
and provides semantic retrieval capabilities for efficient LLM interactions.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

from ..core.logger import logger
from ..services.ai_service_factory import get_ai_service
from ..services.redis_cache import redis_cache
from ..utils.document_normalization import normalize_document_content

def _make_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable format.
    Handles NaN, Timestamp, and other problematic types.
    """
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (np.floating, float)) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


@dataclass
class DocumentChunk:
    """Represents a chunk of document data."""
    chunk_id: str
    file_id: str
    chunk_type: str  # 'financial_data', 'text_content', 'table_data', 'metadata'
    content: Dict[str, Any]
    summary: str
    keywords: List[str]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


# Max rows to store per chunk for tabular data (avoids memory blow-up with large datasets)
MAX_ROWS_PER_TABULAR_CHUNK = 200
HASHING_EMBEDDING_DIM = 1024

# Concurrency limit for LLM summary generation — avoids 429s from the API.
_SUMMARY_SEMAPHORE = asyncio.Semaphore(10)

class DocumentChunkingService:
    """Service for intelligent document chunking and retrieval."""
    
    def __init__(self):
        self.ai_service = get_ai_service()
        # Fixed-space embedding to prevent dimension mismatches between queries/chunks.
        self.vectorizer = HashingVectorizer(
            n_features=HASHING_EMBEDDING_DIM,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )
        
    async def chunk_document(
        self, 
        file_content: Dict[str, Any], 
        filename: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """
        Intelligently chunk a document based on its content type and structure.
        
        Args:
            file_content: The processed file content
            filename: Original filename
            file_type: File extension
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        try:
            if file_type in ['csv', 'xlsx', 'xls']:
                chunks = await self._chunk_tabular_data(file_content, filename)
            elif file_type in ['pdf', 'docx', 'doc', 'txt', 'md', 'rtf', 'odt', 'pptx', 'ppt', 'html', 'xml']:
                chunks = await self._chunk_text_document(file_content, filename)
            elif file_type == 'json':
                chunks = await self._chunk_json_data(file_content, filename)
            else:
                # Fallback: treat as text
                chunks = await self._chunk_text_document(file_content, filename)

            self.attach_embeddings(chunks)
            logger.info(f"[CHUNKING] Created {len(chunks)} chunks for {filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"[CHUNKING] Error chunking {filename}: {e}")
            # Return a single fallback chunk
            fallback = [DocumentChunk(
                chunk_id=self._generate_chunk_id(filename, "fallback"),
                file_id=filename,
                chunk_type="fallback",
                content=file_content,
                summary=f"Complete content of {filename}",
                keywords=["document", "content"]
            )]
            self.attach_embeddings(fallback)
            return fallback
    
    def _resolve_tabular_rows(self, file_content: Any) -> List[Dict[str, Any]]:
        """Resolve row list for tabular chunking (LLM-prepared or agent-fast original_data)."""
        if isinstance(file_content, list):
            return [r for r in file_content if isinstance(r, dict)]

        if not isinstance(file_content, dict):
            return []

        prepared = file_content.get("llm_analysis", {}).get("prepared_data", {})
        structured = prepared.get("structured_data")
        if isinstance(structured, list) and structured:
            if isinstance(structured[0], dict):
                return structured

        original = file_content.get("original_data")
        if isinstance(original, list) and original and isinstance(original[0], dict):
            return original

        content = file_content.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            return content

        return []

    async def _chunk_tabular_data(
        self, 
        file_content: Dict[str, Any], 
        filename: str
    ) -> List[DocumentChunk]:
        """Chunk tabular data (CSV, Excel) by logical sections."""
        chunks = []

        structured_data = self._resolve_tabular_rows(file_content)

        if not structured_data:
            logger.warning(f"[CHUNKING] No structured data found for {filename}")
            return chunks
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(structured_data)
        
        # Chunk by data patterns
        chunks.extend(await self._chunk_by_data_patterns(df, filename))
        
        # Chunk by statistical summaries
        chunks.extend(await self._chunk_by_statistics(df, filename))
        
        # Chunk by time periods (if date columns exist)
        chunks.extend(await self._chunk_by_time_periods(df, filename))
        
        return chunks
    
    async def _chunk_by_data_patterns(
        self, 
        df: pd.DataFrame, 
        filename: str
    ) -> List[DocumentChunk]:
        """Chunk data by identifying patterns and groupings."""
        chunks: list[DocumentChunk] = []
        summary_tasks: list[tuple[int, str, str]] = []  # (chunk_idx, content_sample, chunk_type)

        # Identify numeric columns for financial analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            financial_data_full = df[numeric_cols].to_dict('records')
            total_rows = len(financial_data_full)
            financial_data = financial_data_full[:MAX_ROWS_PER_TABULAR_CHUNK]
            del financial_data_full

            chunks.append(DocumentChunk(
                chunk_id=self._generate_chunk_id(filename, "financial_data"),
                file_id=filename,
                chunk_type="financial_data",
                content=_make_json_serializable({
                    "data": financial_data,
                    "columns": numeric_cols,
                    "row_count": len(financial_data),
                    "total_rows": total_rows,
                }),
                summary="",
                keywords=self._extract_financial_keywords(numeric_cols)
            ))
            summary_tasks.append((len(chunks) - 1, financial_data[:10], "financial_data"))

        # Identify categorical/text columns
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            categorical_data_full = df[text_cols].to_dict('records')
            total_rows = len(categorical_data_full)
            categorical_data = categorical_data_full[:MAX_ROWS_PER_TABULAR_CHUNK]
            del categorical_data_full

            chunks.append(DocumentChunk(
                chunk_id=self._generate_chunk_id(filename, "categorical_data"),
                file_id=filename,
                chunk_type="categorical_data",
                content=_make_json_serializable({
                    "data": categorical_data,
                    "columns": text_cols,
                    "row_count": len(categorical_data),
                    "total_rows": total_rows,
                }),
                summary="",
                keywords=self._extract_categorical_keywords(text_cols)
            ))
            summary_tasks.append((len(chunks) - 1, categorical_data[:10], "categorical_data"))

        # Generate all LLM summaries in parallel
        if summary_tasks:
            tasks = [
                self._generate_chunk_summary(sample, ctype, filename)
                for _, sample, ctype in summary_tasks
            ]
            summaries = await asyncio.gather(*tasks)
            for (idx, _, _), summary in zip(summary_tasks, summaries):
                chunks[idx].summary = summary

        return chunks
    
    async def _chunk_by_statistics(
        self, 
        df: pd.DataFrame, 
        filename: str
    ) -> List[DocumentChunk]:
        """Create chunks based on statistical summaries."""
        chunks = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return chunks
        
        # Generate statistical summary
        stats_summary = df[numeric_cols].describe().to_dict()
        
        # Create summary chunk
        summary_text = f"Statistical summary of {len(numeric_cols)} numeric columns from {filename}"
        
        chunks.append(DocumentChunk(
            chunk_id=self._generate_chunk_id(filename, "statistics"),
            file_id=filename,
            chunk_type="statistics",
            content=_make_json_serializable({
                "statistics": stats_summary,
                "columns": numeric_cols,
                "total_rows": len(df)
            }),
            summary=summary_text,
            keywords=["statistics", "summary", "numeric", "analysis"]
        ))
        
        return chunks
    
    async def _chunk_by_time_periods(
        self, 
        df: pd.DataFrame, 
        filename: str
    ) -> List[DocumentChunk]:
        """Chunk data by time periods if date columns exist."""
        chunks = []
        
        # Look for date columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                date_cols.append(col)
        
        if not date_cols:
            return chunks
        
        # Group by time periods (monthly, quarterly, yearly)
        for date_col in date_cols:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Monthly chunks (cap data per chunk to avoid memory blow-up)
                monthly_groups = df.groupby(df[date_col].dt.to_period('M'))
                for period, group in monthly_groups:
                    period_data_full = group.to_dict('records')
                    total_in_period = len(period_data_full)
                    period_data = period_data_full[:MAX_ROWS_PER_TABULAR_CHUNK]
                    del period_data_full
                    chunks.append(DocumentChunk(
                        chunk_id=self._generate_chunk_id(filename, f"monthly_{period}"),
                        file_id=filename,
                        chunk_type="time_series",
                        content=_make_json_serializable({
                            "data": period_data,
                            "period": str(period),
                            "period_type": "monthly",
                            "date_column": date_col,
                            "total_rows": total_in_period,
                        }),
                        summary=f"Monthly data for {period} from {filename}",
                        keywords=["monthly", "time_series", str(period)]
                    ))
                    
            except Exception as e:
                logger.warning(f"[CHUNKING] Error processing date column {date_col}: {e}")
        
        return chunks
    
    async def _chunk_text_document(
        self,
        file_content: Dict[str, Any],
        filename: str
    ) -> List[DocumentChunk]:
        """Chunk text documents by sections and topics."""
        normalized_document = normalize_document_content(
            file_content,
            filename=filename,
        )
        text_content = (
            normalized_document.get("markdown")
            or normalized_document.get("text")
            or ""
        )

        if not text_content:
            return []

        sections = self._split_text_by_sections(text_content)
        chunks: list[DocumentChunk] = []

        for i, section in enumerate(sections):
            if len(section.strip()) < 50:
                continue
            chunks.append(DocumentChunk(
                chunk_id=self._generate_chunk_id(filename, f"section_{i}"),
                file_id=filename,
                chunk_type="text_content",
                content={
                    "text": section,
                    "section_index": i,
                    "word_count": len(section.split())
                },
                summary="",
                keywords=self._extract_text_keywords(section)
            ))

        # Generate all LLM summaries in parallel
        if chunks:
            tasks = [
                self._generate_chunk_summary(c.content["text"], "text_section", filename)
                for c in chunks
            ]
            summaries = await asyncio.gather(*tasks)
            for chunk, summary in zip(chunks, summaries):
                chunk.summary = summary

        return chunks

    async def _chunk_json_data(
        self, 
        file_content: Dict[str, Any], 
        filename: str
    ) -> List[DocumentChunk]:
        """Chunk JSON data by logical groupings."""
        chunks: list[DocumentChunk] = []

        if isinstance(file_content, dict):
            rows = self._resolve_tabular_rows(file_content)
            if rows:
                file_content = rows

        if isinstance(file_content, list):
            chunk_size = 50
            for i in range(0, len(file_content), chunk_size):
                chunk_data = file_content[i:i + chunk_size]
                chunks.append(DocumentChunk(
                    chunk_id=self._generate_chunk_id(filename, f"batch_{i//chunk_size}"),
                    file_id=filename,
                    chunk_type="json_data",
                    content={
                        "data": chunk_data,
                        "batch_index": i // chunk_size,
                        "total_items": len(chunk_data)
                    },
                    summary="",
                    keywords=["json", "batch", "data"]
                ))

            # Generate all LLM summaries in parallel
            if chunks:
                tasks = [
                    self._generate_chunk_summary(c.content["data"], "json_batch", filename)
                    for c in chunks
                ]
                summaries = await asyncio.gather(*tasks)
                for chunk, summary in zip(chunks, summaries):
                    chunk.summary = summary

        return chunks

    def _split_text_by_sections(self, text: str) -> List[str]:
        """Split text into logical sections."""
        # Split by headers (lines starting with # or all caps)
        sections = re.split(r'\n(?=#{1,6}\s|\n[A-Z][A-Z\s]{10,}\n)', text)
        
        # Also split by double newlines for paragraph breaks
        all_sections = []
        for section in sections:
            paragraphs = section.split('\n\n')
            all_sections.extend([p.strip() for p in paragraphs if p.strip()])
        
        return all_sections
    
    async def _generate_chunk_summary(
        self, 
        content: Any, 
        chunk_type: str, 
        filename: str
    ) -> str:
        """Generate a summary for a chunk using LLM (with Redis cache + concurrency)."""
        # Build content hash for caching
        payload = json.dumps({"content": content, "type": chunk_type, "file": filename}, default=str)
        content_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        cache_key = f"chunk_summary:{content_hash}"

        # Check cache first
        cached = redis_cache.get(cache_key)
        if cached is not None and isinstance(cached, str):
            return cached

        try:
            # Prepare content for LLM
            if isinstance(content, list) and len(content) > 0:
                sample_content = content[:5] if len(content) > 5 else content
                content_str = json.dumps(sample_content, default=str, indent=2)
            elif isinstance(content, str):
                content_str = content[:1000]
            else:
                content_str = str(content)[:1000]

            prompt = (
                "You are summarizing a section from a document. Follow these rules STRICTLY:\n\n"
                "RULE 1: ONLY summarize what is explicitly written in the text below. Do NOT add information that is not in the text.\n"
                "RULE 2: If the text is very short or just a header, provide a brief factual summary based ONLY on what is written.\n"
                "RULE 3: Do NOT say 'I cannot access' or 'I do not have access' - you DO have the text below.\n"
                "RULE 4: Do NOT make up details, dates, or information not in the text.\n"
                "RULE 5: If the text is incomplete or unclear, summarize what IS there, do not apologize.\n\n"
                f"Text from {filename} ({chunk_type}):\n{content_str}\n\n"
                "Provide a concise summary (max 100 words) based ONLY on the text above:"
            )

            # Run sync LLM call in thread pool to avoid blocking the event loop.
            # Semaphore limits concurrency to avoid API rate limits.
            async with _SUMMARY_SEMAPHORE:
                response = await asyncio.to_thread(
                    self.ai_service.chat_completion,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                )

            result = response.strip()

            # Cache the result (TTL: 1 hour — chunk content is static per upload)
            try:
                redis_cache.set(cache_key, result, ttl=3600)
            except Exception:
                pass

            return result

        except Exception as e:
            logger.warning(f"[CHUNKING] Error generating summary: {e}")
            return f"{chunk_type} data from {filename}"
    
    def _extract_financial_keywords(self, columns: List[str]) -> List[str]:
        """Extract financial keywords from column names."""
        keywords = ["financial", "data", "numeric"]
        
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['revenue', 'sales', 'income']):
                keywords.append("revenue")
            if any(term in col_lower for term in ['cost', 'expense']):
                keywords.append("cost")
            if any(term in col_lower for term in ['profit', 'margin']):
                keywords.append("profit")
            if any(term in col_lower for term in ['asset', 'liability']):
                keywords.append("balance_sheet")
        
        return list(set(keywords))
    
    def _extract_categorical_keywords(self, columns: List[str]) -> List[str]:
        """Extract categorical keywords from column names."""
        keywords = ["categorical", "text", "metadata"]
        
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['name', 'title', 'description']):
                keywords.append("descriptive")
            if any(term in col_lower for term in ['category', 'type', 'class']):
                keywords.append("classification")
            if any(term in col_lower for term in ['date', 'time']):
                keywords.append("temporal")
        
        return list(set(keywords))
    
    def _extract_text_keywords(self, text: str) -> List[str]:
        """Extract keywords from text content."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common business/financial terms
        business_terms = [
            'revenue', 'profit', 'cost', 'income', 'expense', 'margin',
            'sales', 'marketing', 'customer', 'product', 'service',
            'financial', 'analysis', 'report', 'data', 'metrics'
        ]
        
        keywords = []
        for term in business_terms:
            if term in text.lower():
                keywords.append(term)
        
        # Add most frequent words (excluding common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'oil', 'sit', 'try'}
        word_freq = {}
        for word in words:
            if word not in common_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add top 5 most frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords.extend([word for word, freq in top_words])
        
        return list(set(keywords))
    
    def _generate_chunk_id(self, filename: str, chunk_type: str) -> str:
        """Generate a unique chunk ID."""
        content = f"{filename}_{chunk_type}_{datetime.now(timezone.utc).isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def find_relevant_chunks(
        self,
        query: str,
        available_chunks: List[DocumentChunk],
        max_chunks: int = 3,
    ) -> List[DocumentChunk]:
        """In-memory fallback for when pgvector DB search is unavailable.

        **Prefer ``DocumentChunkRepository.find_nearest`` for production
        queries** — it uses the pgvector IVFFlat ANN index and runs in
        O(sqrt(n)) inside Postgres.

        This method is kept for backward compatibility and unit tests
        that don't have a live database.
        """
        if not available_chunks:
            return []

        try:
            query_embedding = self._normalize_embedding_dim(
                await self._create_embedding(query)
            )

            similarities = []
            for chunk in available_chunks:
                if chunk.embedding is None:
                    chunk_text = f"{chunk.summary} {' '.join(chunk.keywords)}"
                    chunk.embedding = self._normalize_embedding_dim(
                        await self._create_embedding(chunk_text)
                    )
                else:
                    chunk.embedding = self._normalize_embedding_dim(chunk.embedding)

                sim = cosine_similarity([query_embedding], [chunk.embedding])[0][0]
                similarities.append((chunk, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            relevant = [c for c, _ in similarities[:max_chunks]]

            logger.info(
                f"[CHUNKING] In-memory fallback found {len(relevant)} "
                f"relevant chunks for query: {query[:50]}..."
            )
            return relevant

        except Exception as e:
            logger.error(f"[CHUNKING] Error finding relevant chunks: {e}")
            return available_chunks[:max_chunks]
    
    def attach_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Populate hashing embeddings so pgvector search can find saved chunks."""
        for chunk in chunks:
            if chunk.embedding is not None:
                chunk.embedding = self._normalize_embedding_dim(chunk.embedding)
                continue
            chunk.embedding = self.create_query_embedding(
                self._embedding_source_text(chunk)
            )

    def _embedding_source_text(self, chunk: DocumentChunk) -> str:
        """Build the text used to hash-embed a chunk for similarity search."""
        parts: list[str] = []
        if chunk.summary:
            parts.append(chunk.summary)
        if chunk.keywords:
            parts.append(" ".join(chunk.keywords))
        content = chunk.content
        if chunk.chunk_type == "text_content" and isinstance(content, dict):
            parts.append(str(content.get("text") or ""))
        elif isinstance(content, dict) and "data" in content:
            parts.append(json.dumps(content.get("data")[:3], default=str))
        elif content is not None:
            parts.append(str(content)[:2000])
        return " ".join(p for p in parts if p).strip() or chunk.summary or chunk.chunk_id

    def create_query_embedding(self, text: str) -> List[float]:
        """Public sync helper: create a fixed-dimension embedding for *text*.

        Used by ``SmartDataRetrievalService`` to build the query vector
        before passing it to ``DocumentChunkRepository.find_nearest``.
        """
        return self._normalize_embedding_dim(self._create_hashing_embedding(text))

    async def _create_embedding(self, text: str) -> List[float]:
        """Create deterministic fixed-dimension embedding for text."""
        return self._create_hashing_embedding(text)
    
    def _create_hashing_embedding(self, text: str) -> List[float]:
        """Create fixed-dimension HashingVectorizer embedding."""
        try:
            sparse = self.vectorizer.transform([text])
            return sparse.toarray()[0].astype(float).tolist()
        except Exception as e:
            logger.warning(f"[CHUNKING] Error creating hashing embedding: {e}")
            return [0.0] * HASHING_EMBEDDING_DIM

    def _normalize_embedding_dim(self, embedding: Optional[List[float]]) -> List[float]:
        """
        Enforce fixed embedding dimension for safe cosine similarity.
        Legacy embeddings with different lengths are truncated/padded.
        """
        if not embedding:
            return [0.0] * HASHING_EMBEDDING_DIM
        vec = [float(x) for x in embedding]
        if len(vec) == HASHING_EMBEDDING_DIM:
            return vec
        if len(vec) > HASHING_EMBEDDING_DIM:
            return vec[:HASHING_EMBEDDING_DIM]
        return vec + [0.0] * (HASHING_EMBEDDING_DIM - len(vec))
