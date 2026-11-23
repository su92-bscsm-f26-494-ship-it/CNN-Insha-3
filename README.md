# Image Brightness Classifier — Streamlit

A small Streamlit application that loads a saved Random Forest model, extracts mean red/green/blue (RGB) values from an uploaded image, and displays the model output.

## Important model limitation

The supplied `model.joblib` was inspected and contains a Random Forest classifier with three input features (`mean_r`, `mean_g`, `mean_b`) but only one learned class: `Bright`.

Therefore, it cannot distinguish `Bright` from `Dark`. The app checks for this condition and displays a warning instead of presenting a misleading two-class result.

The supplied training script creates labels from image brightness itself (`Bright` when mean brightness is above 127, otherwise `Dark`). These are brightness labels, not medically verified skin-lesion labels. This project is **not** a skin cancer detector or medical diagnostic system.

## Files

- `app.py` — Streamlit web application.
- `model.joblib` — supplied trained model.
- `requirements.txt` — Python dependencies.
- `train_model.py` — supplied training script (kept for reference).

## Run locally

1. Install Python 3.10 or 3.11.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional but recommended):

   Windows:
   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Start Streamlit:
   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Upload to GitHub

1. Sign in at https://github.com and create a new repository.
2. Give it a name such as `image-brightness-classifier`.
3. Keep the repository public if you want a straightforward public Streamlit deployment.
4. Upload these four project files/folders from this project:
   - `app.py`
   - `model.joblib`
   - `requirements.txt`
   - `README.md`
   - `train_model.py` (optional reference file)
5. Commit the files to the repository's main branch.

You can also use Git from this folder:

```bash
git init
git add app.py model.joblib requirements.txt README.md train_model.py .gitignore
git commit -m "Add Streamlit image classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your actual GitHub details.

## Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in using GitHub.
2. Select **Create app**.
3. Choose your repository, branch (`main`), and main file path (`app.py`).
4. Select **Deploy**.
5. Wait for dependency installation and app startup. Streamlit will provide a public app URL.

If deployment fails, open the app's **Manage app → Logs** and check the first error message.

## Before claiming two-class performance

Retrain the model using real, correctly labelled examples for both classes. Verify that the trained model has both classes in `model.classes_`, and evaluate it on a held-out test set. The current model cannot support a Bright-versus-Dark comparison.
