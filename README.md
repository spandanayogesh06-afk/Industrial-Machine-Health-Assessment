# Industrial Machine Health Assessment

A local Streamlit application for Decision Tree-based machine failure assessment using the AI4I 2020 feature schema. The uploaded machine image is for visual reference only; predictions use the operational parameters entered by the user.

## Setup

1. Use Python 3.13 with the pinned scikit-learn version in `requirements.txt`.
2. Create and activate a virtual environment in this folder:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. The trained artifact is included at `models/machine_failure_model.pkl`. If it is missing, run the final self-contained training cell in `notebooks/Industrial_Machine_Health_Assessment.ipynb` after installing the project dependencies. Do not place untrusted model artifacts here: joblib/pickle loading can execute code.
5. Start the app:

   ```powershell
   streamlit run app.py
   ```

Predictions are unavailable if the trained artifact is absent or invalid. The Streamlit app does not train or synthesize a replacement model.

## Training record

The artifact was trained from the UCI AI4I 2020 Predictive Maintenance Dataset, available at [doi:10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) under CC BY 4.0. The source CSV is `data/ai4i2020.csv`. The notebook's local training cell selects only machine type, the five supported operating parameters, and the `Machine failure` target; auxiliary failure-mode columns are excluded.

Training matches the notebook: stratified 80/20 split with `random_state=42` and `DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)`. On the held-out split, the trained model achieved 0.9295 accuracy and 0.9079 ROC-AUC; failure-class precision was 0.31 and recall was 0.88. These are dataset evaluation results, not guarantees of field performance.

## Expected model artifact

The artifact may be either a model object or a dictionary containing `model` and optionally `features`. Dictionary feature order is honored after validation. A model-only artifact uses this exact input order:

1. `Air temperature [K]`
2. `Process temperature [K]`
3. `Rotational speed [rpm]`
4. `Torque [Nm]`
5. `Tool wear [min]`
6. `Type_L`
7. `Type_M`

Type encoding matches `pd.get_dummies(..., columns=['Type'], drop_first=True)`: H = (0, 0), M = (0, 1), L = (1, 0). The model must expose `predict()` and `predict_proba()`, and class `1` must represent failure.

## Assessment notes

The displayed probability is the loaded model's class-1 output. Risk bands are application triage labels: Low below 20%, Moderate from 20% to below 50%, and High at 50% or above. These are not calibrated reliability guarantees. When the artifact exposes a decision tree, explanations show the actual split rules followed for the submitted input; otherwise, available global feature importance is labeled as such.

PDF reports are generated on demand and downloaded directly; they are not automatically saved to disk. The `reports/` folder is available for local workflow extensions.

The Service Locator builds a location-specific Google Maps search from the user-entered repair phrase and city/postal code. It uses a Maps URL and does not require an API key; Google Maps supplies the live business listings, which the app does not verify.
