# Crop Disease Diagnosis — VLM Demo

A deployed, single-agent Vision-Language Model app for crop disease diagnosis from leaf photos — built as the lightweight, production-facing companion to **[VIDA+PANDA](https://github.com/YOUR_USERNAME/vida-panda-vlm-debate)**, a multi-agent VLM debate framework for the same task.

**[Live demo →](https://crop-disease-vlm-demo-k5wb8fwsxnxd6pxm43bdu9.streamlit.app)** 

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)

---

## What it does

Upload a photo of a crop leaf (or pick a sample) and the app returns:
- A diagnosis from a 59-class taxonomy spanning 13 crop species
- A confidence score
- A short, evidence-grounded explanation citing visual symptoms (lesion color, shape, pattern, affected leaf region)

It runs a single open-weight vision-language model (Llama 4 Maverick via Together.ai) in one pass — a fast, cheap, publicly deployable version of the diagnostic reasoning used in the full multi-agent research framework.

## Why this exists

The [VIDA+PANDA](https://github.com/YOUR_USERNAME/vida-panda-vlm-debate) research framework uses 6 VLM agents across 4 providers in a 3-round debate process to reach a more reliable consensus diagnosis — built for accuracy and research rigor, not real-time public use. This app strips that down to a single model and a single inference call, trading some accuracy for something anyone can try instantly in a browser, with no setup. It demonstrates the same underlying taxonomy and prompting methodology in a form built for actual deployment.

## How it works

```
User uploads/selects image
        │
        ▼
Image downscaled + base64-encoded
        │
        ▼
Single VLM call (Llama 4 Maverick via Together.ai)
   — structured JSON output: label, confidence, reasoning
        │
        ▼
Parsed and rendered as a diagnosis card
```

## Tech stack

- **Streamlit** — UI and deployment
- **Together.ai** — hosted inference for `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- **Pillow** — image preprocessing

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/crop-disease-vlm-demo.git
cd crop-disease-vlm-demo
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your real Together.ai API key

streamlit run app.py
```

Get a free Together.ai API key at [api.together.ai](https://api.together.ai).

## Deploying

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub
2. Connect it on Streamlit Cloud, pointing at `app.py`
3. Add `TOGETHER_API_KEY` under the app's **Settings → Secrets**

## Sample images

The `sample_images/` folder contains 9 example leaf photos (one per class) so the app works immediately without needing your own photos. Sourced from a crop disease image collection; used here for demonstration only.

## Limitations

This is a research demo, not an agronomic decision-making tool. Single-model inference can misdiagnose — the entire point of the multi-agent debate approach in VIDA+PANDA is to catch and correct exactly these kinds of single-model errors through structured disagreement between independent agents.

## Related work

- [VIDA+PANDA](https://github.com/YOUR_USERNAME/vida-panda-vlm-debate) — the full multi-agent debate research framework
- *A Multi-Agent Vision-Language Debate Framework for Zero-Shot Crop Disease Diagnosis* (manuscript in preparation, Frontiers in Plant Science)

## Author

Zayed Al Aghbari — [GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/zayed-al-aghbari)
