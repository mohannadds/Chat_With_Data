# DataChat – Natural Language Data Analysis with Local LLMs (Ollama)

**Ask questions in Arabic or English about your dataset — get instant, accurate answers powered by local large language models.**

A full-stack FastAPI application that lets you upload JSON/CSV-like data and chat with it in natural language (Arabic & English supported), using **100% local Ollama models** — no data leaves your machine.

Perfect for:
- Analysts who want to explore data without writing pandas code
- Arabic-speaking teams needing native-language data insights
- Privacy-focused environments (everything runs locally)
- Quick prototyping and ad-hoc analysis

### Live Demo Example
> "كم عدد الطلبات التي تم تنفيذها في شهر ديسمبر؟"  
> → "عدد الطلبات المنفذة في ديسمبر: ٨٤٧ طلباً"

> "What is the average sales amount by region?"  
> → "Average sales: North → $42,100 | South → $38,700 | ..."

### Key Features
- Natural language → pandas queries (via LangChain + Ollama)
- Full Arabic & English support (auto-detects question language)
- Session-based conversations with context awareness
- In-memory + MongoDB persistence (sessions auto-expire after 24h)
- Smart caching (agents, models, sessions)
- Clean, responsive web interface (HTML + JS + Tailwind-like feel)
- Supports large datasets (tested with 100K+ rows)
- Zero cloud dependency — runs completely offline

### Tech Stack
- FastAPI (backend + API)
- Ollama (local LLMs: `qwen2.5:32b`, `deepseek`, etc.)
- LangChain + pandas-ai agent
- MongoDB (session storage)
- Jinja2 templates + vanilla JS frontend
- Deployable on IIS (wfastcgi), Docker, or bare metal

### Default Model
`qwen2.5:32b-instruct-q2_K` – excellent balance of speed & accuracy, works great in Arabic and English.

You can switch to any model available in your Ollama library.

### Quick Start

```bash
# 1. Make sure Ollama is running with your model pulled
ollama pull qwen2.5:32b-instruct-q2_K

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start MongoDB (local or Docker)

# 4. Run the app
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
