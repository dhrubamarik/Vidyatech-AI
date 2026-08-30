import os
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq, APIError
from pypdf import PdfReader
from docx import Document

class _ScopeStore:
    """One isolated FAISS index + its parallel bookkeeping lists. Every
    scope (e.g. one per user, one per community) gets its own instance,
    so nothing indexed under one scope can ever be retrieved from
    another -- that isolation is the whole point of keying by scope
    instead of keeping one process-wide index."""

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []   # parallel to documents, kept so we can rebuild the index on delete
        self.chunk_owner: List[str] = []         # owner label (doc id) each chunk belongs to, for surgical removal


class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        # Isolated per-scope vector stores. A "scope" is an opaque string
        # the caller controls -- typically "user:<id>" so each student's/
        # faculty member's uploads only ever get retrieved for that same
        # person, or "community:<id>" for material meant to be shared
        # with everyone in one community. Nothing is ever searched across
        # scopes.
        self.stores: Dict[str, _ScopeStore] = {}

        api_key = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
        self.groq_client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-120b"

        self.agent_prompts = {
            "tutor": "You are an expert Academic Tutor. Explain concepts clearly with step-by-step examples. Directly address the user's specific question — do not go off-topic even if extra context is provided.",
            "quiz": "You are a Quiz Master. Generate 3 multiple-choice questions (MCQs) with an answer key about the user's requested topic.",
            "summarizer": "You are a Summarizer. If the provided context is relevant to the user's question, summarize the relevant parts in clear, structured bullet points. If the context does NOT relate to the user's question, say so plainly instead of summarizing unrelated material — never summarize content the user didn't ask about.",
            "doubt_solver": "You are a Precision Doubt Solver. Answer the user's specific question directly and concisely using the context."
        }
        # FAISS L2 distance above which a "match" is considered too weak to
        # actually be relevant (embeddings are roughly unit-normalized, so
        # this sits around a 0.5 cosine-similarity cutoff). Prevents
        # unrelated queries from being answered using whatever's nearest.
        self.RELEVANCE_THRESHOLD = 1.1

    def _store(self, scope: str) -> _ScopeStore:
        """Get (creating if needed) the isolated store for this scope."""
        if scope not in self.stores:
            self.stores[scope] = _ScopeStore(self.dimension)
        return self.stores[scope]

    def clear_vector_store(self, scope: str):
        """Wipe only this scope's documents -- e.g. one user's knowledge
        base -- leaving every other scope untouched."""
        self.stores.pop(scope, None)

    def extract_text_from_file(self, file_path: str) -> str:
        text = ""
        try:
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception as e:
            raise ValueError(f"Failed to parse document: {str(e)}")
            
        return text.strip()

    def process_and_index_document(self, file_path: str, scope: str, owner: str = None) -> int:
        """Extract, chunk, and embed a document into the given scope's
        *own* index. `scope` isolates this from every other scope's
        documents (e.g. "user:42"). `owner` is a label (e.g. the document's
        DB id) used later to remove exactly this document's chunks from
        that scope's index without touching anyone else's."""
        raw_text = self.extract_text_from_file(file_path)
        
        if not raw_text:
            raise ValueError("No extractable text found. The document may be empty, image-only, or password-protected.")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(raw_text)
        
        if not chunks:
            return 0
        
        owner = owner or os.path.basename(file_path)
        store = self._store(scope)
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True).astype("float32")
        store.index.add(embeddings)
        store.documents.extend(chunks)
        store.embeddings.extend(list(embeddings))
        store.chunk_owner.extend([owner] * len(chunks))
        
        return len(chunks)

    def remove_document(self, scope: str, owner: str) -> int:
        """Remove every chunk belonging to `owner` within this scope's
        index (matched against the label passed into
        process_and_index_document) and rebuild that scope's FAISS index
        from what's left. Returns how many chunks were removed. Other
        scopes are never touched."""
        store = self.stores.get(scope)
        if store is None:
            return 0

        keep_idx = [i for i, o in enumerate(store.chunk_owner) if o != owner]
        removed = len(store.chunk_owner) - len(keep_idx)

        store.documents = [store.documents[i] for i in keep_idx]
        store.embeddings = [store.embeddings[i] for i in keep_idx]
        store.chunk_owner = [store.chunk_owner[i] for i in keep_idx]

        store.index = faiss.IndexFlatL2(self.dimension)
        if store.embeddings:
            store.index.add(np.vstack(store.embeddings).astype("float32"))

        return removed

    def summarize_text(self, raw_text: str, max_chars: int = 8000) -> str:
        """One-shot AI summary used for community notes. Truncates very
        long documents to keep the request fast and inexpensive; good
        enough for a shareable at-a-glance summary."""
        excerpt = raw_text[:max_chars]
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.agent_prompts["summarizer"]},
                    {"role": "user", "content": f"Summarize this document for students:\n\n{excerpt}"}
                ],
                model=self.model,
                temperature=0.3
            )
            return response.choices[0].message.content
        except APIError as e:
            return f"AI summary unavailable: {str(e)}"

    def retrieve_context(self, query: str, scope: str, top_k: int = 3) -> Dict[str, Any]:
        """Search only within `scope`'s own index -- documents indexed
        under any other scope are never visible here."""
        store = self.stores.get(scope)
        if store is None or store.index.ntotal == 0 or len(store.documents) == 0:
            return {"context": "", "found": False, "reason": "No documents indexed for this scope."}
        
        if len(query.split()) > 15:
            top_k = min(5, store.index.ntotal)
        else:
            top_k = min(top_k, store.index.ntotal)
            
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = store.index.search(query_vec, top_k)

        # Nothing indexed is actually close to this query -- treat it as
        # "not found" rather than answering with whatever's nearest.
        if len(distances[0]) == 0 or distances[0][0] > self.RELEVANCE_THRESHOLD:
            return {"context": "", "found": False, "reason": "No sufficiently relevant matches found."}

        retrieved_chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist <= self.RELEVANCE_THRESHOLD and 0 <= idx < len(store.documents):
                retrieved_chunks.append(store.documents[idx])
        
        if not retrieved_chunks:
            return {"context": "", "found": False, "reason": "No relevant matches found."}
            
        return {"context": "\n\n".join(retrieved_chunks), "found": True, "reason": "Success"}

    def coverage_for_topics(self, scope: str, topics: List[str]) -> List[Dict[str, Any]]:
        """Used by the syllabus heatmap: for each topic, how well does
        *this scope's* indexed knowledge base actually cover it? Buckets
        into GREEN (strong match), YELLOW (weak/partial match), RED
        (nothing close)."""
        store = self.stores.get(scope)
        results = []
        for topic in topics:
            if store is None or store.index.ntotal == 0:
                results.append({"topic": topic, "status": "RED", "confidence": 0})
                continue
            query_vec = self.embedder.encode([topic], convert_to_numpy=True).astype("float32")
            top_k = min(3, store.index.ntotal)
            distances, _ = store.index.search(query_vec, top_k)
            best = float(distances[0][0]) if len(distances[0]) else 999.0

            if best <= self.RELEVANCE_THRESHOLD * 0.55:
                status = "GREEN"
            elif best <= self.RELEVANCE_THRESHOLD:
                status = "YELLOW"
            else:
                status = "RED"
            confidence = max(0, round(100 - (best / self.RELEVANCE_THRESHOLD) * 100))
            results.append({"topic": topic, "status": status, "confidence": min(confidence, 100)})
        return results

    def generate_agent_response(self, query: str, scope: str, agent_type: str = "doubt_solver") -> Dict[str, Any]:
        retrieval_res = self.retrieve_context(query, scope)
        context = retrieval_res["context"]
        
        if not retrieval_res["found"]:
            context_prompt = (
                "CRITICAL INSTRUCTION: The requested information was NOT found in the uploaded document. "
                "Do NOT answer general knowledge questions. Politely inform the user:"
                "'This query is outside the scope of your uploaded document.'"
            )
        else:
            context_prompt = (
                f"Context from Document (only use this if it actually helps answer the question below; "
                f"if it doesn't, ignore it and answer from the question alone):\n{context}"
            )

        system_prompt = self.agent_prompts.get(agent_type.lower(), self.agent_prompts["doubt_solver"])
        prompt = f"{context_prompt}\n\nUser Question:\n{query}"
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3
            )
            return {
                "status": "success",
                "response": response.choices[0].message.content,
                "has_context": retrieval_res["found"]
            }
        except APIError as e:
            return {
                "status": "error",
                "response": f"LLM API Error: {str(e)}",
                "has_context": False
            }

    def raw_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """A bare LLM call with no retrieval step, for features that build
        their own context (recap generator, quiz generator, cheat sheet)."""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=temperature
            )
            return response.choices[0].message.content
        except APIError as e:
            return f"LLM API Error: {str(e)}"

rag_engine = RAGPipeline()