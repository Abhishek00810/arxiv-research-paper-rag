# ArXiv Research Assistant — Build Plan
**By: Abhishek | Goal: Portfolio project + interview asset**

---

## What You're Building

An **Agentic RAG system** that answers complex questions over ArXiv AI/ML papers.

**Why ArXiv:**
- Nobody owns this space cleanly
- Genuinely hard multi-hop questions
- Directly relevant to target companies (E2B, Modal, Sarvam)
- Can demo it on papers about Firecracker, vLLM, RAG itself
- No "just use Claude Code" objection
- Live at research.abhid.me after deployment

---

## Final Architecture

```
User Query (FastAPI)
        ↓
[LangGraph Agent]
        ↓
[query_analyzer]     — simple vs complex classification
        ↓
[retriever]          — hybrid BM25 + dense search → RRF merge → BGE reranker
        ↓
[relevance_grader]   — LLM judges: sufficient / insufficient
        ↓ insufficient (max 2 retries)    ↓ sufficient
[query_rewriter]                     [generator]
→ back to retriever                  → answer + citations
                                          ↓
                                     FastAPI response
                                     Langfuse trace logged
                                     RAGAS evaluable
```

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Vector DB | Qdrant (Docker local) | Free, hybrid search native, production-ready |
| Embedding model | BGE-M3 via fastembed | SOTA open weights, CPU-optimized for Mac |
| Reranker | BGE-Reranker via sentence-transformers | Cross-encoder, accurate, open weights |
| Sparse retrieval | BM25 (built into Qdrant) | Catches exact term matches dense misses |
| Merging | Reciprocal Rank Fusion | Simple, effective, no extra model needed |
| Agent framework | LangGraph | Stateful cyclic loops, not linear chains |
| LLM | Groq (LLaMA 3.3 70B) | Free tier, 800+ tok/sec, OpenAI-compatible |
| LLM backup | MiniMax M2.5 API | Very generous free tier if Groq rate limits |
| Observability | Langfuse (cloud free tier) | Traces every agent node visually |
| Evals | RAGAS | Measures faithfulness, precision, recall |
| API layer | FastAPI | Async, auto-docs, production standard |
| PDF parsing | PyMuPDF (fitz) | Best for ArXiv PDFs |
| Chunking | LangChain SemanticChunker | Meaning boundaries not character count |

---

## Pre-Saturday Setup (30 mins tonight)

Get accounts + verify environment. No coding yet.

```bash
# Verify Python
python3 --version   # need 3.10+

# Verify Docker
docker --version

# Verify Git
git --version
```

**Sign up for (all free):**
- groq.com → get API key
- langfuse.com → get public key + secret key
- huggingface.co → create account (for model downloads)
- cloud.qdrant.io → account ready (for deployment later, not needed this weekend)

**Download 5-10 ArXiv papers as PDFs (your test corpus):**
- Firecracker (2020): arxiv.org/abs/2003.09058
- vLLM / PagedAttention: arxiv.org/abs/2309.06180
- RAG original paper: arxiv.org/abs/2005.11401
- Attention Is All You Need: arxiv.org/abs/1706.03762
- DeepSeek R1: arxiv.org/abs/2501.12948
- Flash Attention: arxiv.org/abs/2205.14135
- LoRA: arxiv.org/abs/2106.09685
- LangGraph / ReAct: arxiv.org/abs/2210.03629

Save all to: `~/arxiv-agent/data/papers/`

---

## Saturday Morning — Ingestion Pipeline (3-4 hrs)

### Step 1: Project Setup

```bash
mkdir arxiv-agent && cd arxiv-agent
python -m venv venv && source venv/bin/activate
mkdir -p data/papers src

git init
echo "venv/" > .gitignore
echo ".env" >> .gitignore
```

```bash
pip install langchain langchain-community langgraph \
    qdrant-client fastembed \
    langchain-groq sentence-transformers \
    ragas langfuse fastapi uvicorn \
    pymupdf python-dotenv datasets
```

```
# .env file
GROQ_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

```bash
# Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### Step 2: PDF Parser (src/parser.py)

```python
# Goal: extract clean text from ArXiv PDFs
# Use PyMuPDF (fitz)
# Handle: title, abstract, sections separately
# Store metadata: paper_id, title, section, page_number
```

Write this yourself. Key decisions:
- Extract section headers separately (useful for chunking context)
- Store paper title + arxiv ID as metadata
- Handle multi-column ArXiv layout (PyMuPDF handles this)

### Step 3: Semantic Chunker (src/chunker.py)

```python
# Goal: split text on meaning boundaries not character count
# Use LangChain SemanticChunker
# NOT RecursiveCharacterTextSplitter
# Why: semantic boundaries preserve context better
#      a fixed 512 token chunk can split mid-concept
```

Key decisions you need to make and understand:
- Breakpoint threshold type: `percentile` vs `standard_deviation`
- How to preserve section context in chunk metadata
- Minimum chunk size (avoid tiny orphan chunks)

### Step 4: Embedding + Indexing (src/indexer.py)

```python
# Goal: embed chunks and store in Qdrant with both
#       dense vectors AND sparse (BM25) vectors

# fastembed handles BGE-M3 on CPU efficiently
# Qdrant supports hybrid indexing natively

# Collections setup:
# - dense vector: 1024 dims (BGE-M3 output size)
# - sparse vector: BM25 via FastEmbed sparse model
# - payload: chunk text, paper_id, title, section, page
```

First time you run this: BGE-M3 downloads (~2.2GB). Just wait.

**By end of Saturday morning:** Run ingestion on all 8 papers. Verify chunks are stored in Qdrant. Open Qdrant dashboard at localhost:6333/dashboard and manually inspect some chunks.

---

## Saturday Afternoon — Hybrid Retrieval + Reranker (3-4 hrs)

### Step 5: Hybrid Retriever (src/retriever.py)

Three stages chained:

```python
# Stage 1: Parallel search
# BM25 sparse search → top 50 results
# Dense vector search → top 50 results
# Run both, get two ranked lists

# Stage 2: Reciprocal Rank Fusion
# Merge two lists into one
# Formula: score = 1/(rank_in_dense + 60) + 1/(rank_in_bm25 + 60)
# Chunks appearing in both lists get boosted
# Result: single ranked list of top 50

# Stage 3: BGE-Reranker (cross-encoder)
# Take top 50 from RRF
# BGE-Reranker reads query + each chunk TOGETHER
# Scores each chunk 0-1 for relevance
# Return top 5 highest scored chunks
```

**Why this order matters:**
- Dense + BM25 are fast → use for coarse filtering (top 50)
- Reranker is slow but accurate → use only on shortlist (top 50 → top 5)
- Running reranker on full index would be too slow

**Test this manually before moving on:**
```python
results = retriever.search("how does PagedAttention manage KV cache")
for r in results:
    print(r.score, r.payload['text'][:100])
```

Do the results make sense? If not, debug here. Don't carry broken retrieval into the agent layer.

---

## Sunday Morning — LangGraph Agent (3-4 hrs)

### Step 6: Agent State + Nodes (src/agent.py)

**State schema first:**
```python
from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    query: str
    original_query: str
    retrieved_chunks: List[dict]
    relevance_decision: str    # "sufficient" or "insufficient"
    relevance_reason: str
    retry_count: int
    final_answer: str
    sources: List[str]
```

**Five nodes — write each separately, test each separately:**

**Node 1: query_analyzer**
```python
# Input: query
# Output: classification (simple/complex) + reasoning
# Use Groq LLM
# Prompt: "classify this query as simple (single-hop)
#          or complex (multi-hop, requires connecting
#          information from multiple sources)"
# Simple queries skip decomposition
# Complex queries get flagged for multi-source retrieval
```

**Node 2: retriever**
```python
# Input: query (may be rewritten)
# Output: top 5 chunks from hybrid retrieval pipeline
# Just calls your src/retriever.py from Saturday
```

**Node 3: relevance_grader**
```python
# This is the CORE agentic node
# Input: query + retrieved chunks
# Output: "sufficient" or "insufficient" + reason

# Prompt must be TIGHT:
# "Given ONLY these retrieved chunks and nothing else,
#  can you answer this question completely and accurately?
#  Answer: sufficient or insufficient
#  Reason: one sentence explaining why"

# Common mistake: prompt too lenient → always says sufficient
# Fix: emphasize "ONLY these chunks, no other knowledge"
```

**Node 4: query_rewriter**
```python
# Input: original query + why retrieval failed
# Output: rewritten query targeting the gap

# Prompt:
# "The query '{query}' failed to retrieve good context.
#  Reason: {relevance_reason}
#  Rewrite the query to specifically target
#  the missing information."

# Max 2 retries enforced in graph edges
```

**Node 5: generator**
```python
# Input: query + sufficient chunks
# Output: answer + list of sources (paper title + section)

# Prompt must enforce citation:
# "Answer using ONLY the provided context.
#  End with Sources: [paper title, section] for each chunk used"
```

**Graph edges (the loop logic):**
```python
def route_after_grading(state: AgentState) -> str:
    if state["relevance_decision"] == "sufficient":
        return "generator"
    elif state["retry_count"] >= 2:
        return "generator"    # generate best effort answer
    else:
        return "query_rewriter"

# This is where the agentic loop lives
# Without this conditional edge → just a linear pipeline
```

---

## Sunday Afternoon — FastAPI + Evals + README (2-3 hrs)

### Step 7: FastAPI wrapper (src/api.py)

```python
@app.post("/query")
async def query(request: QueryRequest):
    result = agent.invoke({"query": request.question})
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retries": result["retry_count"],
        "trace_url": result.get("langfuse_trace_url")
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Step 8: RAGAS Evaluation (src/evals.py)

**Write 15 test questions manually. This is 2-3 hours of work but it's the most important part.**

Question types to include:
```
Simple factual (5 questions):
→ "What is PagedAttention?"
→ "How does Firecracker achieve isolation?"
→ "What problem does LoRA solve?"

Medium complexity (5 questions):
→ "How does Flash Attention reduce memory usage
    compared to standard attention?"
→ "What are the tradeoffs of using LoRA vs full fine-tuning?"

Complex multi-hop (5 questions):
→ "How do the memory management approaches in
    Firecracker and PagedAttention differ conceptually?"
→ "What is the relationship between the original
    attention mechanism and Flash Attention's optimization?"
```

**Run evals TWICE:**
```
Round 1 — Baseline (before your full pipeline):
→ Disable reranker, use fixed chunking, single dense search
→ Record: faithfulness, context_precision, answer_relevancy

Round 2 — Your full pipeline:
→ Semantic chunking + hybrid search + reranker + agent loop
→ Record same metrics

Document the delta. This is your resume story.
```

### Step 9: README.md

Structure:
```markdown
# ArXiv Research Assistant

## What it does
## Architecture diagram (ASCII is fine)
## Stack
## Eval Results
| Metric | Baseline | Full Pipeline |
|--------|----------|---------------|
| Faithfulness | 0.xx | 0.xx |
| Context Precision | 0.xx | 0.xx |
| Answer Relevancy | 0.xx | 0.xx |

## Key Design Decisions
(explain WHY you made each choice — this is for interviews)

## Running Locally
(exact commands, no ambiguity)
```

---

## Extension Roadmap (After Weekend)

Priority order based on your goals (fastest to land + high ceiling):

```
Week 2 (small additions, big signal):
→ HyDE (Hypothetical Document Embeddings)
   LLM generates hypothetical answer → embed that → search
   Improves vague query retrieval significantly
   Measure RAGAS improvement, add to README
   
→ Streaming responses (FastAPI SSE)
   Stream tokens as they generate
   Production UX signal

Week 3 (makes it genuinely impressive):
→ Multi-hop query decomposition
   Break complex questions into sub-questions
   Retrieve for each, synthesize final answer
   Run RAGAS on complex questions specifically

Week 4-5 (puts it in different league):
→ GraphRAG
   Extract entities + relationships from papers
   Build knowledge graph (Neo4j or NetworkX)
   Enable lineage tracing: "how did concept X evolve"
   This is frontier RAG work in 2026

Month 2 (closes the full stack story for Modal):
→ Deploy quantized Qwen 3.5 7B with vLLM
   Self-hosted inference instead of Groq API
   Measure tokens/sec, cost per query
   Connects retrieval pipeline to infra serving story
```

---

## Deployment (Week 2)

```
Qdrant: Qdrant Cloud free tier (1GB, enough for 500+ papers)
FastAPI: Render.com free tier or Railway
Domain: research.abhid.me (point subdomain via Cloudflare)
Langfuse: cloud.langfuse.com free tier
Total cost: $0
```

---

## Resume Bullet (After Weekend)

> Built an agentic RAG system for querying 500+ ArXiv AI/ML papers — hybrid BM25 + dense retrieval with BGE reranking, self-evaluating LangGraph control loop with query rewriting on retrieval failure. Improved faithfulness from 0.xx → 0.xx and context precision from 0.xx → 0.xx vs naive baseline. Deployed at research.abhid.me.

---

## Interview Answers This Project Gives You

**"Walk me through your RAG architecture"**
→ Explain hybrid search + RRF + reranker + agent loop
→ Show Langfuse trace of agent deciding to retry

**"Why hybrid search over just dense vectors?"**
→ Dense misses exact term matches (function names, paper titles)
→ BM25 catches these, RRF merges both signal types

**"Why a cross-encoder reranker if you already have dense search?"**
→ Bi-encoder (dense): encodes query and doc separately, fast, approximate
→ Cross-encoder: reads query + doc together, slow, much more accurate
→ Use dense for coarse filtering (top 50), reranker for precision (top 5)

**"How do you prevent the agent loop from running forever?"**
→ max_retries=2 hardcoded in routing edge
→ After 2 retries: generate best-effort answer with low confidence signal

**"How did you measure quality?"**
→ RAGAS: faithfulness, context precision, answer relevancy
→ 15 manually written ground truth Q&A pairs
→ Baseline vs full pipeline comparison
→ Specific number improvements

---

## Rules While Building

```
1. Write every line yourself first
2. Stuck 30+ mins → ask AI to EXPLAIN, not write
3. Working code → paste to AI for review
4. Never copy-paste AI code without retyping
5. After each session: write down what broke + how you fixed it
   These become your interview stories
```

---

## Key Concepts to Understand Deeply Before Saturday

Spend 1 hour reading tonight — not coding, just understanding:

1. **Why LangGraph over LangChain**: cyclic graphs vs linear chains
2. **Bi-encoder vs cross-encoder**: the fundamental reranker tradeoff
3. **BM25 intuition**: TF-IDF based, why it catches exact matches
4. **RRF formula**: simple math, understand it not just use it
5. **RAGAS faithfulness**: LLM-as-judge, what it's actually measuring

---

## The Goal

By Sunday evening you have:
- Working system end to end
- RAGAS scores (baseline + improved)
- Langfuse traces showing agent reasoning
- FastAPI running locally
- Clean README with architecture + numbers
- Ready to deploy Monday

By week 2:
- Live at research.abhid.me
- Resume updated
- Outreach starts — don't wait for perfect