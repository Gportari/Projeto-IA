# ML Workspace — Model Training & Comparison

A Flask-based web app to register machine learning models via JSON configs, train and compare them with live progress, and test predictions. The UI uses a small design system and a reusable Vue sidebar component.

## Highlights
- Model registry from `models_json/*.json` rendered in `Models.html`.
- Training and comparison in `ModelComparison.html` with server-side polling.
- Aggregated training progress for Random Forest across all estimators.
- Centralized theme tokens and sticky sidebar (`static/theme.css`).

## Quick Start
- Prerequisites: Python 3.11+ (tested with 3.13) and `pip`.
- Install dependencies: `pip install -r requirements.txt`
- Run the server: `python app.py`
- Open: `http://127.0.0.1:5000/`

## Project Structure
```
app.py                      # Flask app and API routes
model_registry.py           # Reads and groups model configs
models_json/                # JSON configs per model
static/
  components/sidebar-nav.js # Vue-based sidebar
  model_comparison.js       # Frontend training/test logic
  models.js                 # Field lists and UI helpers
  theme.css                 # Design system and theme tokens
templates/
  Models.html               # Model registry page
  ModelComparison.html      # Training & comparison page
trainingModels/             # Model implementations
  LogisticRegression.py
  KNN.py
  SVM.py
  DecisionTree.py
  RandomForest.py           # Keras-based ensemble
tests/diagnostic_tests.py   # Simple diagnostics
requirements.txt            # Python dependencies
```

## Pages
**Models**
- Lists all registered models from `/api/models`.
- Shows model type and its parameters parsed from JSON.

**Model Comparison**
- Choose a model + dataset + training params.
- Starts training and polls `/api/train/status/<job_id>` every second.
- Displays epoch, total_epochs, metrics (when complete), and ETA.

## Model Configs
- Location: `models_json/*.json`.
- Typical fields (Random Forest example):
```
{
  "nome_teste": "Teste Random Forest 1",
  "nome_modelo": "RandomForestClassifier",
  "rf_n_estimators": 10,
  "rf_max_depth": "None",
  "rf_learning_rate": 0.01,
  "rf_epochs": 50,
  "rf_batch_size": 32
}
```
- Notes:
  - `rf_max_depth` accepts numeric or the string `"None"`.
  - If a numeric field is given as `"None"`, `"null"`, or empty, the backend applies safe defaults.

## Training Parameters
- Passed from the UI and consumed by the backend:
  - `epochs`: total epochs per estimator (`int`).
  - `batch_size`: batch size (`int`).
  - `validation_split`: fraction for validation (`float`, default `0.2`).
  - `test_split`: train/test split (`float`, default `0.2`).

## API
- `GET /api/models`
  - Returns grouped models (type, filename, params) for the UI.

- `POST /api/train/start`
  - Starts a job. Returns `{ job_id }`.
  - Payload example:
```
{
  "model": { "type": "rf", "config": "RandomForest_example_1.json" },
  "dataset": { "features": [[...], [...]], "labels": [1, -1] },
  "training_params": { "epochs": 50, "batch_size": 32, "validation_split": 0.2, "test_split": 0.2 }
}
```

- `GET /api/train/status/<job_id>`
  - Live job status: `epoch`, `total_epochs`, `loss`, `accuracy`, `eta_seconds`, `status`.

- `POST /api/train/cancel/<job_id>`
  - Requests cancellation.

- `GET /api/train/result/<job_id>`
  - Final metrics and summary when `status=completed`.

- `POST /api/test/<job_id>`
  - Run predictions on a trained model.
  - Payload: `{ features: [...], labels?: [...] }`.
  - Returns `predictions` and optional metrics (accuracy, precision, recall, f1, confusion matrix).

## Implementation Notes
- Random Forest progress:
  - Training uses one `keras.Model.fit()` per estimator.
  - Backend aggregates progress so epochs don’t appear to “reset” between estimators.
  - `total_epochs` is reported as `epochs * rf_n_estimators`.

- Parameter parsing robustness:
  - Numeric fields gracefully handle `"None"`, `"null"`, or empty strings.

- UI/Design system:
  - Sticky sidebar and theme toggle via `static/theme.css` and `static/components/sidebar-nav.js`.

## Development
- Add a new model in `trainingModels/` with:
  - `create_model_from_config(config, training_params)` returning a model instance.
  - `train_and_evaluate(model_config, X_train, X_test, y_train, y_test, training_params)` returning `(results, model)`.
- Update `model_registry.py` and `static/models.js` if new parameters are introduced.
- Keep changes consistent, minimal, and focused.

## Testing
- Run diagnostics: `python tests/diagnostic_tests.py`
- Validate training and progress in `Model Comparison` after starting the server.

## Troubleshooting
- Random Forest: `invalid literal for int() with base 10: 'None'`
  - Fixed by robust parsing in `trainingModels/RandomForest.py`.
  - Ensure JSON uses `"None"` for optional numeric fields or provide a number.
- Epochs “resetting” on Random Forest
  - Resolved by aggregated progress in `app.py`.
  - `total_epochs` equals `epochs * n_estimators`; progress is continuous.
- If UI doesn’t load
  - Check the server output and confirm `http://127.0.0.1:5000/` is reachable.
  - Verify dependencies via `pip install -r requirements.txt`.

## Notes
- This project prioritizes clarity and simple defaults; avoid overcomplicating configs.