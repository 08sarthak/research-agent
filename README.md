# Skin Report Assistant

A Gradio app that:
- accepts a skin image + user inputs,
- runs a dummy `predict_disease` function (hardcoded output),
- sends inputs + prediction to an LLM (`sonar-pro`) to generate an informational report.

## Project Structure

- `client/app.py` - Gradio UI and app flow
- `client/predict_disease.py` - dummy prediction function
- `core/agent.py` - LLM prompt + report generation
- `core/schemas.py` - Pydantic schemas

## Requirements

- Python 3.10+
- A valid `PERPLEXITY_API_KEY`

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt -r client/requirements.txt
```

Create a `.env` file in the project root with:

```env
PERPLEXITY_API_KEY=your_real_key_here
PERPLEXITY_MODEL=sonar-pro
```

## Run

```bash
python client/app.py
```

Open the local URL shown in terminal (typically `http://127.0.0.1:7860`).

## Notes

- The disease prediction is currently hardcoded in `client/predict_disease.py`.
- This app is informational only and not medical advice.

