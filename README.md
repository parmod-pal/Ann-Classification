# ANN Classification — Customer Churn

Short description
This project trains an Artificial Neural Network (ANN) to predict customer churn using the "Churn_Modelling.csv" dataset. It includes preprocessing, model training with TensorFlow/Keras, saved preprocessing artifacts, TensorBoard logging, and a notebook (`experiments.ipynb`) to reproduce experiments.

Repository layout
- experiments.ipynb — main notebook: load data, preprocess, split/scale, build/train ANN, save artifacts, TensorBoard cells.
- app.py — optional script to run training/inference (may import TensorFlow).
- requirements.txt — pinned and required Python packages.
- model.h5 — saved Keras model (created by training).
- scaler.pkl — saved StandardScaler (created by training).
- label_encoder_gender.pkl — saved LabelEncoder for gender.
- onehot_encoder_geo.pkl — saved OneHotEncoder for Geography.
- logs/fit — TensorBoard logs (created after training).
- Churn_Modelling.csv — input dataset (not included in repo, add to project root).

Requirements
- macOS (tested)
- Python 3.10+ recommended
- See requirements.txt (this project pins tensorflow==2.15.0)

Quick setup (recommended: venv)
1. Create virtual environment and activate:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Upgrade pip and install requirements:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. If protobuf / TensorFlow compatibility errors occur, install the compatible protobuf:
   ```bash
   python -m pip install "protobuf==3.20.3"
   ```
4. Verify installation:
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
   python -m pip show protobuf
   ```

How to run / reproduce
1. Place `Churn_Modelling.csv` in the project root.
2. Open `experiments.ipynb` in VS Code or Jupyter and run cells top-to-bottom. Key steps:
   - Load CSV and drop irrelevant columns.
   - Encode categorical features (LabelEncoder for Gender, OneHotEncoder for Geography).
   - Split: train_test_split(X, y, test_size=0.2, random_state=42)
   - Scale numeric features using StandardScaler (fit on training data only).
   - Build Keras Sequential model and train with EarlyStopping and TensorBoard callbacks.
   - Save artifacts: `model.h5`, `scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`.
3. Or run `app.py` from terminal (if implemented for training/inference):
   ```bash
   python3 app.py
   ```

Notes about preprocessing
- StandardScaler is used to standardize numeric features (zero mean, unit variance). Fit only on training set to avoid leakage.
- One-hot encoded geography produces extra columns; the notebook concatenates these back into the feature matrix.
- Encoders and scaler are saved with pickle so they can be reused at inference.

Example inference snippet
```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl','rb') as f:
    le_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl','rb') as f:
    ohe_geo = pickle.load(f)

# Build a sample in the same feature order used for training.
# Replace the numeric order with the exact order in your notebook.
sample_num = np.array([[600, 1, 40, 3, 60000.0, 1, 1, 1, 50000.0]])  # numeric features example
gender_encoded = le_gender.transform(['Male'])
geo_vec = ohe_geo.transform([['France']]).toarray()
X = np.hstack([sample_num, gender_encoded.reshape(-1,1), geo_vec])  # adjust ordering to match training
X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)
print("churn_probability:", float(pred[0][0]))
```

Troubleshooting
- AttributeError: 'FieldDescriptor' object has no attribute 'is_repeated' — usually protobuf/TensorFlow mismatch. Fix:
  ```bash
  python -m pip install "protobuf==3.20.3"
  ```
  Reinstall TensorFlow in the same env if necessary.
- If training is slow or fails on macOS, ensure you installed the CPU-only TensorFlow build appropriate for your platform or configured Apple acceleration (tensorflow-metal) if using an M1/M2 GPU.
- If you need newer protobuf for other projects, isolate this repo in its own virtual environment.

Best practices and suggestions
- Persist a full preprocessing + model pipeline (sklearn Pipeline or custom wrapper) to avoid mismatch between training and serving.
- Add unit checks / assertions after split:
  ```python
  assert len(X_train) + len(X_test) == len(X)
  ```
- Do not fit scalers/encoders on full dataset before splitting.

What I tested / known issues
- The notebook currently uses train_test_split unpacking in correct order (X_train, X_test, y_train, y_test). Ensure you run cells in order.
- requirements.txt pins tensorflow==2.15.0. If install fails on your machine, try a different TF build or a different Python version (see TensorFlow official install notes).

License
Add an appropriate license file (e.g., MIT) and contributor information.

Contact / contribution
Open an issue or PR with improvements, diagnostics, or environment notes.
