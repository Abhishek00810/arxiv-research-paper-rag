# CLAUDE.md — ArXiv Research Assistant

## Who I Am
Abhishek — infrastructure/platform engineer, 1.5 years experience.
Strong background in Go, systems programming, Firecracker microVMs.
Building this as a portfolio project targeting AI infra companies
(E2B, Modal, Daytona, Sarvam).

## What This Project Is
An **Agentic RAG system** for querying ArXiv AI/ML research papers.
Not a tutorial follow-along — a real portfolio piece with evals,
observability, and a live deployment at research.abhid.me.

## My Goals With This Project
1. Learn the AI pipeline layer hands-on (retrieval, agents, evals)
2. Ship something measurable with RAGAS scores on resume
3. Understand every component deeply enough to defend in interviews
4. Get it deployed and live within 2 weeks

---

## How to Help Me

### The Most Important Rule
**Help me understand, not just ship.**

I need to be able to defend every decision in interviews.
When I ask for help:
- Explain WHY before showing HOW
- Point out what I got wrong in my approach
- Ask me questions that reveal if I actually understand
- Don't just fix my code — make sure I understand the fix

### What Good Help Looks Like
```
Me: "My reranker returns high scores for irrelevant chunks"
You: Explain what a cross-encoder actually does,
     why it might score irrelevant chunks highly,
     what to look at in my code,
     then review my code and point to the issue
```

### What Bad Help Looks Like
```
Me: "Write me the LangGraph agent"
You: *writes entire agent*
→ I learn nothing, can't defend it in interviews
→ Don't do this
```

### When I Ask You to Write Code
- Write it WITH explanation of every decision
- Point out alternatives I should know about
- Tell me what will likely break and why
- Ask me to explain it back before moving on

---

## Architecture

```
User Query (FastAPI POST /query)
        ↓
[LangGraph Agent — src/agent.py]
        ↓
[query_analyzer node]
→ classifies: simple (single-hop) or complex (multi-hop)
        ↓
[retriever node]
→ calls src/retriever.py
→ BM25 sparse search (top 50) + dense vector search (top 50)
→ Reciprocal Rank Fusion merges both lists
→ BGE-Reranker cross-encoder scores top 50 → returns top 5
        ↓
[relevance_grader node]
→ LLM judges if chunks are sufficient to answer query
→ Returns: "sufficient" or "insufficient" + reason
        ↓
[conditional edge]
→ sufficient → generator node
→ insufficient + retry_count < 2 → query_rewriter node
→ insufficient + retry_count >= 2 → generator (best effort)
        ↓
[query_rewriter node] (if insufficient)
→ LLM rewrites query targeting the gap
→ loops back to retriever
        ↓
[generator node]
→ LLM generates answer from retrieved context ONLY
→ Returns answer + source citations
        ↓
FastAPI response
Langfuse trace logged automatically
```

---

## Tech Stack

| Component | Tool | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Vector DB | Qdrant | Docker latest |
| Embeddings | BGE-M3 via fastembed | latest |
| Reranker | BGE-Reranker-v2-m3 via sentence-transformers | latest |
| Agent framework | LangGraph | latest |
| LLM primary | Groq (llama-3.3-70b-versatile) | API |
| LLM backup | MiniMax M2.5 | API |
| Observability | Langfuse | cloud free tier |
| Evals | RAGAS | latest |
| API | FastAPI + uvicorn | latest |
| PDF parsing | PyMuPDF (fitz) | latest |
| Chunking | LangChain SemanticChunker | latest |

---

## Project Structure

```
arxiv-agent/
├── CLAUDE.md                  ← you are here
├── .env                       ← API keys (never commit)
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── papers/                ← ArXiv PDFs go here
├── src/
│   ├── parser.py              ← PDF text extraction
│   ├── chunker.py             ← semantic chunking
│   ├── indexer.py             ← embed + store in Qdrant
│   ├── retriever.py           ← hybrid search + reranker
│   ├── agent.py               ← LangGraph agent + all nodes
│   ├── evals.py               ← RAGAS evaluation suite
│   └── api.py                 ← FastAPI wrapper
├── tests/
│   └── test_retriever.py
└── evals/
    └── ground_truth.json      ← 15 manually written Q&A pairs
```

---

## Environment Variables

```bash
# .env
GROQ_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# optional backup LLM
MINIMAX_API_KEY=
```

---

## Key Design Decisions (Know These For Interviews)

### Why Semantic Chunking over Fixed Size
Fixed 512-token chunks split mid-concept.
SemanticChunker splits on meaning boundaries — a complete idea stays in one chunk.
Critical for technical papers where a concept spans variable length.

### Why Hybrid Search (BM25 + Dense)
Dense vectors understand semantic meaning but miss exact matches.
BM25 catches exact term matches (paper titles, author names, function names, acronyms).
RRF merges both signal types — chunks appearing in both lists get boosted.
Neither alone is as good as both combined.

### Why Reranker After Hybrid Search
Dense search = bi-encoder (encodes query and doc separately) — fast, approximate.
Reranker = cross-encoder (reads query + doc together) — slow, accurate.
Too slow to run on full index. Run hybrid search for coarse top-50, reranker for precise top-5.

### Why LangGraph over LangChain
LangChain = linear pipeline. Cannot loop back.
LangGraph = cyclic graph. Can loop: retrieve → grade → rewrite → retrieve again.
The self-evaluating loop IS the agentic behavior. Requires cycles.

### Why Max 2 Retries
Prevents infinite loops on genuinely unanswerable queries.
After 2 retries: generate best-effort answer with low confidence signal in response.
Cost control — each retry = additional LLM calls.

### Why RAGAS for Evals
LLM-as-judge approach. Measures:
- Faithfulness: is answer grounded in retrieved context? (detects hallucination)
- Context Precision: are retrieved chunks relevant? (measures retrieval quality)
- Context Recall: did we find all important information?
- Answer Relevancy: does answer address the question?
Run twice: naive baseline vs full pipeline. Delta is the resume story.

---

## AgentState Schema

```python
from typing import TypedDict, List

class AgentState(TypedDict):
    query: str                  # current query (may be rewritten)
    original_query: str         # never changes — for context
    retrieved_chunks: List[dict] # from retriever
    relevance_decision: str     # "sufficient" or "insufficient"
    relevance_reason: str       # why grader decided this
    retry_count: int            # max 2
    final_answer: str           # from generator
    sources: List[str]          # paper title + section per chunk used
```

---

## Qdrant Collection Setup

```python
# Two vector types per chunk:
# 1. Dense: BGE-M3 output, 1024 dimensions, cosine similarity
# 2. Sparse: BM25 via fastembed sparse model

# Payload per chunk:
{
    "text": str,           # chunk content
    "paper_id": str,       # arxiv ID e.g. "2003.09058"
    "title": str,          # paper title
    "section": str,        # section heading
    "page": int,           # page number
    "chunk_index": int     # position in document
}
```

---

## LLM Prompts (Starting Points — Expect to Iterate)

### Query Analyzer
```
Classify this research question as "simple" or "complex".

Simple: answerable from a single source, factual, single-hop
Complex: requires connecting information from multiple papers,
         comparing approaches, tracing concept evolution

Question: {query}

Respond with JSON: {{"type": "simple"|"complex", "reason": "..."}}
```

### Relevance Grader
```
You are evaluating retrieved context for a research question.

Question: {query}

Retrieved chunks:
{chunks}

Can these chunks ALONE answer the question completely and accurately?
Do not use any external knowledge. Only what is in these chunks.

Respond with JSON: {{"decision": "sufficient"|"insufficient", "reason": "one sentence"}}
```

### Query Rewriter
```
A retrieval attempt failed to find relevant context.

Original question: {original_query}
Failed attempt: {query}
Why it failed: {relevance_reason}

Rewrite the query to specifically target the missing information.
Be more specific. Use different terminology if needed.
Return only the rewritten query, nothing else.
```

### Generator
```
Answer the research question using ONLY the provided context.
Do not use any knowledge outside these chunks.
If the context is insufficient, say so explicitly.

Question: {query}

Context:
{chunks}

Provide a clear, accurate answer.
End with:
Sources: [list each paper title and section used]
```

---

## RAGAS Test Questions (15 Required)

Write ground truth answers yourself after reading the papers.
Save to evals/ground_truth.json.

**Simple (5):**
1. What is PagedAttention and what problem does it solve?
2. How does Firecracker achieve strong isolation between VMs?
3. What is the core idea behind LoRA fine-tuning?
4. What problem does Flash Attention solve compared to standard attention?
5. What is retrieval-augmented generation (RAG)?

**Medium (5):**
6. How does Flash Attention reduce memory complexity?
7. What are the tradeoffs between LoRA and full fine-tuning?
8. How does Firecracker's design differ from traditional containers?
9. What is the role of the KV cache in LLM inference?
10. How does DeepSeek R1 use reinforcement learning?

**Complex multi-hop (5):**
11. How do the memory management approaches in Firecracker and PagedAttention address similar problems at different layers?
12. What is the conceptual relationship between the original attention mechanism and Flash Attention's optimization?
13. How does LoRA's approach to parameter efficiency relate to the scaling challenges described in the Attention Is All You Need paper?
14. Compare how Firecracker and gVisor approach the isolation vs performance tradeoff differently.
15. How does the RAG approach complement the limitations of the transformer architecture described in Attention Is All You Need?

---

## Eval Baseline vs Full Pipeline

Run evals twice. Record both. The delta is the story.

**Baseline config (deliberately worse):**
- Fixed 512 token chunking (RecursiveCharacterTextSplitter)
- Dense search only (no BM25)
- No reranker (return raw top-5 from dense search)
- No agent loop (single pass)

**Full pipeline config:**
- Semantic chunking
- Hybrid search (BM25 + dense + RRF)
- BGE-Reranker
- LangGraph self-evaluating loop

---

## Common Bugs to Watch For

```
Qdrant:
→ "Collection already exists" on re-run → delete collection first
   or add upsert logic

BGE-M3:
→ First download: 15-20 mins, looks frozen → just wait
→ fastembed caches to ~/.cache/fastembed after first download

LangGraph:
→ "State key not found" → TypedDict schema mismatch
   check every node returns ALL state keys
→ Loop not cycling → check conditional edge function
   must return string matching exact node name

Relevance Grader:
→ Always returns "sufficient" → prompt too lenient
   add: "Only answer sufficient if ALL parts of the
   question are answerable from ONLY these chunks"
→ Always returns "insufficient" → prompt too strict
   check chunk quality first, may be retrieval problem

RAGAS:
→ Score 0.0 on everything → ground_truth format wrong
   must be List[List[str]] for contexts
→ Needs OpenAI API key by default for judge LLM
   override with: evaluator_llm=your_groq_llm

Groq:
→ Rate limit → add exponential backoff retry
→ "Model not found" → check exact model string:
   "llama-3.3-70b-versatile"
```

---

## What I Want to Ship

**By end of Sunday:**
- Working end-to-end locally
- RAGAS scores for both baseline and full pipeline
- Langfuse traces showing agent loop reasoning
- FastAPI running at localhost:8000
- Clean README with architecture + numbers

**By end of week 2:**
- Deployed at research.abhid.me
- HyDE added and measured
- Resume updated
- Outreach started

---

## What I Do NOT Want

- Code I can't explain
- Over-engineered solutions
- Abstractions I don't understand yet
- To skip debugging by asking you to fix everything
- Perfect code at the cost of understanding

---

## How to Run

```bash
# Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Activate environment
source venv/bin/activate

# Ingest papers
python src/indexer.py

# Run evals
python src/evals.py

# Start API
uvicorn src.api:app --reload --port 8000

# Test
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PagedAttention?"}'
```

---

## Interview Questions This Project Answers

1. Walk me through your RAG architecture
2. Why hybrid search over just vector search?
3. Why a cross-encoder reranker if you have dense search?
4. How does your agent decide when retrieval is sufficient?
5. How do you prevent infinite loops?
6. How did you measure quality?
7. What's the difference between faithfulness and context precision?
8. What would you change to scale this to 10M documents?
9. Why LangGraph over writing the loop yourself?
10. What was the hardest bug you hit and how did you fix it?