# Slooze Take-Home — Software Engineer (AI)

Two AI agents in a single Python package, sharing one LLM client (AWS Bedrock → Claude Opus 4.7):

- **Challenge A — Web Search Agent.** Natural-language query → DuckDuckGo search → page fetch → Claude (via AWS Bedrock) synthesises a grounded answer with cited sources.
- **Challenge B — PDF RAG Agent.** PDF in → page-aware text extraction → chunk + local embeddings → FAISS index → retrieval-augmented Q&A and summarisation, again grounded by Claude.

Both flow through one CLI: `python -m agent.cli <command>`.

---

## Setup

Tested on Python 3.11, Windows + Linux.

```bash
git clone <this-repo>
cd software-engineering-AI
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Configure AWS credentials with Bedrock access (any of the standard mechanisms work — `aws configure`, env vars, or an SSO profile):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export BEDROCK_REGION=us-east-1                            # default
export BEDROCK_MODEL_ID=global.anthropic.claude-opus-4-7 # default; any Claude inference profile works
```

The model ID is a Bedrock inference profile — Opus 4.7 is the default; any other Claude profile works with no code change.

---

## Run

### Challenge A — Web Search Agent

```bash
python -m agent.cli search "What are the latest specs in MacBook this year?"
```

Sample output:

```
Answer:
The latest MacBook Air (2025) ships with the Apple M4 chip — 10-core CPU,
8/10-core GPU, 16-core Neural Engine, 120GB/s memory bandwidth — a 13.6"
Liquid Retina display, 16GB unified memory (up to 32GB), Wi-Fi 6E, two
Thunderbolt 4 ports, MagSafe 3, and up to 18 hours of video playback.

Sources:
- https://support.apple.com/en-in/122209
```

### Challenge B — PDF RAG Agent

```bash
# Summarise the document
python -m agent.cli pdf-summarize path/to/doc.pdf

# Ask one question
python -m agent.cli pdf-ask path/to/doc.pdf "What methodology was used in the study?"

# Interactive chat
python -m agent.cli pdf-chat path/to/doc.pdf
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) into the local Hugging Face cache.

---

## Architecture

```
                ┌─────────────────────────────────────┐
                │            agent.llm                │
                │  BedrockClaude → invoke_model()     │
                └─────────────────────────────────────┘
                          ▲                 ▲
                          │                 │
   ┌──────────────────────┴──┐   ┌──────────┴──────────────────────┐
   │  agent.web_search_agent │   │      agent.pdf_rag_agent        │
   │  ─────────────────────  │   │  ─────────────────────────────  │
   │  query                  │   │  PDF                            │
   │   ↓ ddgs (DuckDuckGo)   │   │   ↓ pypdf (page-aware extract)  │
   │  top-N results          │   │  page text                      │
   │   ↓ httpx (4s timeout)  │   │   ↓ window chunker (800/120)    │
   │  page text              │   │  chunks                         │
   │   ↓ trim to budget      │   │   ↓ MiniLM-L6-v2 (local, 384d)  │
   │  context bundle         │   │  embeddings                     │
   │   ↓ Claude synthesise   │   │   ↓ FAISS IndexFlatIP (cosine)  │
   │  Answer + Sources       │   │  retrieve top-k → Claude        │
   │                         │   │   ↓                             │
   │                         │   │  Answer + cited excerpt nums    │
   └─────────────────────────┘   └─────────────────────────────────┘
                          ▲                 ▲
                          └─────── CLI ─────┘
```

### Files

| Path | Role |
|------|------|
| `agent/llm.py` | Bedrock-Claude client. Single point of model/region config; both agents go through this. |
| `agent/web_search_agent.py` | Challenge A. Search → fetch → trim → synthesise. |
| `agent/pdf_rag_agent.py` | Challenge B. Extract → chunk → embed → FAISS → retrieve → synthesise. |
| `agent/cli.py` | One CLI with `search`, `pdf-summarize`, `pdf-ask`, `pdf-chat` subcommands. |
| `requirements.txt` | Pinned-floor deps. |

---

## Design decisions and trade-offs

**LLM via Bedrock, not the Anthropic public API.**
Lets the same code run inside an AWS account with no API-key rotation, IAM-scoped, and bills against the org's existing Bedrock quota. The wrapper exposes a single `complete(system, user)` method so swapping providers is one file.

**DuckDuckGo for web search.**
No API key, no quota negotiation — the take-home spec lists DDG as a valid choice. The result schema is identical to what Tavily/SerpAPI return, so swapping in a higher-recall provider is a single file change. Trade-off: DDG returns thinner snippets than Tavily, which is why the agent fetches pages directly and trims to a context budget.

**Page fetch is best-effort.**
`httpx` with a 4-second timeout, follow-redirects on, fall back to the search snippet if the page errors or 4xx/5xxs. This keeps the agent robust on flaky sites without retries blowing up latency.

**Local embeddings (MiniLM-L6-v2) over Bedrock embeddings.**
Self-contained, free, and fast on CPU. Quality is sufficient for single-document QA at the scales this challenge implies. For production with cross-document corpora I would move to Bedrock Titan Embed v2 or Voyage and persist the FAISS index — `BedrockClaude` already proves the embedding side of the same pattern.

**FAISS in-memory.**
The challenge centers on a single PDF, so persistence buys nothing. The `PdfRag` constructor builds the index once; `Chroma`-backed persistence is a drop-in replacement if needed.

**Window chunking, not semantic chunking.**
800 chars with 120-char overlap. Predictable, deterministic, no extra deps. A semantic chunker (sentence-aware with token budget) would help on heterogeneous PDFs (slides, multi-column papers), and is the obvious next iteration.

**Summary path uses spread-sampling, not retrieval.**
Summaries should reflect the whole document, not the densely-similar chunks a retriever would surface. The agent samples chunks evenly across the document instead, so the LLM sees the full arc.

**Strict citation prompts.**
Both system prompts forbid invented URLs / sources. The web agent cites by URL, the PDF agent cites by excerpt number. Citations are checked qualitatively in test runs — no automated grader, but the prompt + source-bundling structure makes hallucinated cites rare.

**No silent retries / no chain-of-thought scaffolding.**
A single Claude call per question. Easier to reason about cost, latency, and failure modes than a multi-step agent loop. The pieces are there if a tool-calling loop is needed later — the LLM client returns text directly so structured-output (e.g. tool use) can be added without rewriting either agent.

---

## What I'd add next

- **Tool-calling loop on Challenge A** (Claude decides when to search vs. answer from context, follow-up queries on weak retrieval).
- **Hybrid retrieval** (BM25 + dense) for Challenge B — embedding-only retrieval misses literal matches on names/numbers.
- **Persisted index** with deduped chunk IDs for multi-document Q&A.
- **Eval harness** — fixed query/answer pairs scored against ground truth for both agents, run in CI.
