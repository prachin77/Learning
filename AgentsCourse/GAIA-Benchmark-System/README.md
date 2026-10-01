# GAIA Agent — Final Assignment

A LangGraph + Gemini 2.0 Flash agent for the [HuggingFace Agents Course](https://huggingface.co/learn/agents-course) final assignment.

## 🏗️ Architecture

- **Framework:** LangGraph (ReAct Agent)
- **LLM:** Google Gemini 2.0 Flash (FREE)
- **UI:** Gradio (deployed on HuggingFace Spaces)

### Tools
| Tool | Purpose |
|------|---------|
| `web_search` | DuckDuckGo search for factual questions |
| `wikipedia_search` | Direct Wikipedia API lookup |
| `visit_webpage` | Scrape full webpage content |
| `execute_python_file` | Run attached Python files |
| `read_excel_file` | Parse Excel/spreadsheet files |
| `get_youtube_transcript` | Extract YouTube video transcripts |
| Gemini Vision | Analyze images (chess boards, etc.) |
| Gemini Audio | Transcribe and understand audio files |

## 🚀 Setup

### 1. Clone & Install
```bash
cd "4. FinalAssignment-GAIAAgent"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Set API Key
```bash
# Copy the example env file
cp .env.example .env
# Edit .env and add your Gemini API key
# Get a FREE key from: https://aistudio.google.com/
```

### 3. Run Locally
```bash
python app.py
```

## 🌐 Deploy to HuggingFace Spaces

1. Duplicate the [template space](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)
2. Replace files with this codebase
3. Add `GEMINI_API_KEY` as a Secret in Space Settings
4. Your Space will auto-build and run

## 📊 GAIA Benchmark

- 20 Level-1 questions from the GAIA validation set
- Scoring: EXACT MATCH (answer must match ground truth exactly)
- Target: ≥30% (6/20) for certificate, aiming for 75%+ (15/20)
