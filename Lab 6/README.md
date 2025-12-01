# How to Run the Program

Option 1:
```
docker compose up --build
```

Option 2:
```
docker build -t dockerfile .
docker run -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" dockerfile
```

Option 3:
```
pip install -r src/requirements.txt
streamlit run src/dashboard.py
```

## Local dashboard

The UI lets you tune Inverse regularization strength (C), max iterations, test split, and random seed. Click **Train model** to start; a spinner shows while training. The classification report and confusion matrix render as tables; accuracy appears as a metric. Artifacts save to `./artifacts` by default.

## Docker

Build and run the image directly:

```
docker build -t dockerfile .
docker run -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" dockerfile
```

The `artifacts` bind mount ensures saved weights appear in your project root.

## Docker Compose

`docker-compose.yaml` builds the image and binds artifacts:

```
docker compose up --build
```

This exposes `http://localhost:8501` and mounts `./artifacts` to `/app/artifacts`.
