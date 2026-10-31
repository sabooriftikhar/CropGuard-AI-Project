# CropGuard AI

Plant Leaf Disease Classification using EfficientNetB0 transfer learning.

## Required repository structure

Keep these files in the repository root:

- `app.py`
- `requirements.txt`
- `model.keras`
- `class_names.json`
- `metrics.json` (optional)

Optional plots:
- `cropguard_accuracy.png`
- `cropguard_loss.png`
- `cropguard_confusion_matrix.png`

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

Install dependencies and launch:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push the project files to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select your repository, branch, and `app.py`.
4. Deploy and inspect logs if dependency installation fails.

## Add your project links later

In `app.py`, update:
- `PROJECT_REPORT_URL` for your Word report
- `PRESENTATION_URL` for your PowerPoint
- `GITHUB_REPO_URL` for your GitHub repository

## Disclaimer

This is an educational prototype. Validation results do not guarantee performance on real-world field images. Do not rely on predictions alone for agricultural diagnosis or treatment decisions.
