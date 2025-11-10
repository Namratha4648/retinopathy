# Diabetic Retinopathy — Streamlit Demo

This small Streamlit app provides a UI for running the EfficientNet-B2 model from this repo and shows Grad-CAM explainability overlays.

Files added:

- `streamlit_app.py` — main Streamlit UI. Upload an image or pick a sample and press "Predict & Explain".
- `model_utils.py` — helper functions to load the model and compute Grad-CAM.
- `requirements.txt` — Python dependencies.

Quick start (PowerShell on Windows):

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Make sure you have your trained model saved at `models/best_model_finetuned.pth` (the notebook in this repo saves there).

4. Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

Notes:

- If you used a different training stack (e.g. `efficientnet_pytorch`), the TimM model loader in `model_utils.py` is tolerant (it loads state dict with `strict=False`). If you hit missing keys or mismatches, either re-save the checkpoint with compatible keys or adjust `model_utils.load_model` to recreate the original model class.
- Grad-CAM uses the last block / conv head depending on the model; you can change the target layer in the sidebar.

If you'd like, I can also:

- Add a small unit test for `model_utils` (fast on CPU with a random tensor).
- Add a dockerfile or a GitHub Actions workflow to run the app.
