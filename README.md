# RAG Project

## 1. Set up Python environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure your OpenRouter API key

Copy `.env.example` to `.env` and replace the placeholder with your **new** OpenRouter API key:

```env
OPENROUTER_API_KEY=your_new_openrouter_api_key
```

Never commit or share `.env`. It is excluded by `.gitignore`.

## 3. Run the scripts

```bash
python BasicAIRAg_model.py
python RagTrainedModel.py
```

`RagTrainedModel.py` reads `TechNova_Employee_Handbook_Detailed.pdf` from the same folder as the script.
The first run may download the embedding model and create the local Chroma database.

## GitHub security note

The previously exposed OpenRouter key must be revoked in the OpenRouter dashboard. Create a new key and keep it only in your local `.env` file or your deployment provider's environment settings. This archive intentionally contains no `.git` history.
