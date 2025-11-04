# MLflow Breast Cancer Classification Lab

This lab is an end-to-end exercise in experiment tracking with MLflow using the scikit-learn Breast Cancer Wisconsin dataset. You will load and explore the data, build baseline and tuned models, register them in the MLflow Model Registry, and experiment with both Spark-based and REST-based inference pathways.

---

## 1. Learning Objectives

- Understand how to structure an ML experiment so that every run is reproducible and auditable.
- Use MLflow Tracking to log parameters, metrics, artifacts, and models for scikit-learn estimators.
- Promote improved runs through the MLflow Model Registry workflow (Staging → Production).
- Serve a registered model locally and send real-time scoring requests.
- Optionally integrate the model with PySpark for batch inference jobs.

---

## 2. Tools and Components

- **Python** 3.10+ (the provided `requirements.txt` assumes CPython)
- **scikit-learn** for data access and model training
- **MLflow** for experiment tracking, model registration, and serving
- **Hyperopt** for hyperparameter search
- **PySpark** (optional section) for distributed inference
- **Matplotlib / Seaborn** for exploratory analysis
- **JDK 8/11/17** required by PySpark components

---

## 3. Dataset Overview

The lab uses `load_breast_cancer(as_frame=True)` from scikit-learn:

- **Samples**: 569 observations of breast masses captured via digitized fine needle aspirate (FNA).
- **Features**: 30 numeric predictors describing geometric properties (radius, texture, smoothness, symmetry) captured for mean, standard error, and worst-case values.
- **Label**: Binary target (`0 = malignant`, `1 = benign`). The notebook maps these integers to human-readable class names to aid interpretation.

The dataset ships with scikit-learn, so no external downloads are needed.

---

## 4. Repository Map

| Path | Purpose |
| --- | --- |
| `main.ipynb` | Notebook implementing the entire lab. Run cells sequentially. |
| `requirements.txt` | All Python dependencies (MLflow, scikit-learn, Hyperopt, PySpark, etc). |
| `mlruns/` | Local MLflow tracking directory populated after runs. Ignore unless inspecting artifacts. |
---

## 5. Environment Setup

### 5.1 Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

### 5.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This ensures `setuptools` (for `pkg_resources`), `mlflow`, `hyperopt`, and `pyspark` are present.

### 5.3 Configure Java for PySpark (Required for Spark Cells)

Spark only supports JDK 8, 11, or 17. On macOS with SDKMAN!:

```bash
sdk use java 17.0.17-tem          # or any compatible 8/11/17 release
export JAVA_HOME=$SDKMAN_CANDIDATES_DIR/java/current
export PATH="$JAVA_HOME/bin:$PATH"
java -version                     # verify you see the chosen JDK
```

Re-run these exports every time you open a fresh shell prior to starting Jupyter.

---

## 6. Running the Lab Notebook

1. **Launch Jupyter** from the configured environment:
   ```bash
   jupyter notebook
   ```
2. Open `main.ipynb` and execute each cell from top to bottom. The notebook is structured into the following sections:
   - **Setup & Data Loading**: Imports dependencies, loads the breast cancer dataset, and performs feature engineering (dropping the target, renaming columns).
   - **Exploratory Data Analysis**: Generates class balance plots and feature boxplots to compare benign vs malignant distributions.
   - **Baseline Model (`RandomForestClassifier`)**:
     - Splits data into train/validation/test sets with `train_test_split`.
     - Trains a small Random Forest within `mlflow.start_run`.
     - Logs parameters (e.g., `n_estimators`), metrics (`auc`), and artifacts (model + signature).
   - **Model Registry Promotion**:
     - Registers the baseline model under `breast_cancer_classifier`.
     - Pauses to allow the registration job to complete before transitioning stages.
   - **Hyperparameter Tuning with Hyperopt**:
     - Defines a search space for another estimator (e.g., XGBoost) and logs each candidate run as a child run.
     - Selects the best run, registers it, and transitions it to Production, archiving the baseline.
   - **Batch & Real-Time Inference Options**:
     - Demonstrates how to create a Spark UDF for distributed scoring.
     - Shows how to send REST requests to a locally served MLflow model.

> **Tip**: Notebook cells that launch servers (`mlflow ui`, model serving) intentionally block execution. Run those commands in a separate terminal unless the lab instructions state otherwise.

---

## 7. Using the MLflow Tracking UI

1. In a separate terminal (with the same virtual environment activated), start the UI:
   ```bash
   mlflow ui --port 5001
   ```
2. Open `http://localhost:5001` in a browser.
3. Inspect the runs created by the notebook. Compare metrics from the baseline and tuned experiments.
4. Explore artifacts such as the logged `conda.yaml`, the model pickle, and the inferred signature.

_Backgrounding with `!mlflow ui &` directly inside the notebook is blocked by IPython; always start the UI externally or use `subprocess.Popen` if you must automate it._

---

## 8. Model Registry Walkthrough

The notebook demonstrates the following flow for the model named `breast_cancer_classifier`:

1. **Baseline Registration**: Registers the initial Random Forest model (`version 1`) and transitions it to `Production`.
2. **Hyperopt Search**: Runs a tuned model experiment. Each candidate becomes a child run.
3. **Promote Improved Model**:
   - Register the best-performing child run as `version 2`.
   - Archive `version 1`, promote `version 2` to `Production`.
4. **Load the Production Model**:
   ```python
   model = mlflow.pyfunc.load_model("models:/breast_cancer_classifier/production")
   auc = roc_auc_score(y_test, model.predict(X_test))
   ```
   This serves as a sanity check that the production stage is updated.

---

## 9. Serving and Inference

### 9.1 Local Model Serving (Manual Step)

```bash
mlflow models serve \
  --env-manager=local \
  -m models:/breast_cancer_classifier/production \
  -h 0.0.0.0 \
  -p 5002
```

Keep this process running while you send requests. Terminate it with `Ctrl+C` when done.

### 9.2 REST Prediction Example

Within the notebook or a separate script:

```python
import requests

payload = {"dataframe_split": X_test.to_dict(orient="split")}
resp = requests.post("http://localhost:5002/invocations", json=payload)
print(resp.json())
```

### 9.3 Spark UDF Batch Inference

If you run the optional PySpark section:

```python
from pyspark.sql import SparkSession
import mlflow.pyfunc

spark = SparkSession.builder.appName("MLflow Integration").getOrCreate()
apply_model_udf = mlflow.pyfunc.spark_udf(spark, "models:/breast_cancer_classifier/production")
# new_data = spark.read.format("csv").load(table_path)
# scored = new_data.withColumn("prediction", apply_model_udf(*new_data.columns))
```

Make sure your Java configuration is correct before creating the `SparkSession`. Stop the session at the end with `spark.stop()` to free resources.

---

## 10. Validation Checklist

After completing the lab, confirm the following:

- [ ] `main.ipynb` runs top-to-bottom without errors using the configured environment.
- [ ] MLflow UI shows at least two runs (baseline and Hyperopt search runs) with logged `auc` metrics.
- [ ] Model Registry contains `breast_cancer_classifier` with multiple versions and the latest set to `Production`.
- [ ] Local serving endpoint returns predictions for the provided test payload.
- [ ] Optional: Spark UDF section executes successfully (if JDK configured).

---

Happy experimenting!
