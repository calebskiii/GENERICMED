OFFLINE HANDWRITING RECOGNITION APP

FILES
- app.py: Streamlit user interface and CNN-TCN-CTC inference code
- requirements.txt: Python packages needed by the app
- cnn_tcn_gated_fixedlr_v2.pth: your trained checkpoint (copy this file into this folder)

SETUP (Windows PowerShell)
1. Install Python 3.10 or 3.11 (recommended for PyTorch compatibility).
2. Open PowerShell in this folder.
3. Create and activate a virtual environment:
   py -m venv venv
   .\venv\Scripts\Activate.ps1
4. Install dependencies:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
5. Copy cnn_tcn_gated_fixedlr_v2.pth into this folder.
6. Start the app:
   streamlit run app.py
7. Open the local address Streamlit prints (usually http://localhost:8501).

CHECKPOINT
By default, the app expects cnn_tcn_gated_fixedlr_v2.pth in the same folder as app.py.
If it is stored elsewhere, enter its full path in the app's sidebar.

IMPORTANT NOTES
- The app uses the architecture and 79-character vocabulary from the supplied notebook.
- It uses the notebook's custom-image preprocessing: grayscale, Otsu thresholding,
  resize to height 128 while preserving aspect ratio, and normalization to [-1, 1].
- It loads the checkpoint once. It does not train or fine-tune the model.
- The app is intended for one-line English handwriting input. Long multi-line text is
  not automatically segmented into separate lines.
