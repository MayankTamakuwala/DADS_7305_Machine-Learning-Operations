# BERT BIO Tagging API for SPO Triplet Extraction

A FastAPI-based web service that extracts Subject-Predicate-Object (SPO) triplets from unstructured text using a fine-tuned BERT model with BIO tagging. This API is designed for knowledge graph construction and semantic relationship extraction.

## Overview

This project provides a RESTful API that processes natural language text and extracts structured relationships in the form of SPO triplets. The system uses a pre-trained BERT model fine-tuned for BIO (Beginning-Inside-Outside) tagging to identify subjects, predicates, and objects within text, then forms meaningful triplets for knowledge graph construction.


## API Endpoints

### Health Check
- **GET** `/` - Check API health status
- **Response**: `{"status": "healthy"}`

### Triplet Extraction
- **POST** `/predict` - Extract SPO triplets from input text
- **Request Body**: 
  ```json
  {
    "input_str": "Apple Inc. announced a new product launch in the technology sector."
  }
  ```
- **Response**:
  ```json
  {
    "response": [
      {
        "subject": "Apple Inc.",
        "predicate": "announced",
        "object": "new product launch"
      }
    ]
  }
  ```

## Project Structure

```
Lab 1/
├── src/
│   ├── main.py              # FastAPI application and endpoints
│   ├── predict.py           # Core prediction logic and BIO tagging
│   ├── model.py             # BERT model definition
│   ├── load_weights.py      # Model weight downloading utility
├── model/
│   └── BERT_BIO_Tagging_model.pth  # Pre-trained model weights
├── assets/                  # Documentation images
├── requirements.txt         # Python dependencies
├── run.py                   # Application runner
└── README.md                # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Lab 1"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy English model** (if not already installed)
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Application

### Development Mode
```bash
python run.py
```

### Using uvicorn directly
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs

## Model Information

### BERT BIO Tagging Model
- **Task**: Token-level classification for 7 BIO tags
- **Tags**: O, B-SUB, I-SUB, B-PRED, I-PRED, B-OBJ, I-OBJ
- **Model Weights**: Downloaded from [Hugging Face](https://huggingface.co/MayankTamakuwala/BERT_BIO_Tagger)

### Training Process
For detailed information about the model training process, dataset preparation, and evaluation metrics, please visit the original research repository on my GitHub Account:

**🔗 [Tagging Models for SPO Pairs Extraction](https://github.com/MayankTamakuwala/Tagging_Models_For_SPO_Pairs_Extraction)**

This repository contains:
- Complete training notebooks and scripts
- Dataset preparation and annotation tools
- Model evaluation and comparison with Viterbi algorithm
- Detailed results and performance metrics

## Usage Examples

### Using cURL
```bash
# Health check
curl -X GET "http://localhost:8000/"

# Extract triplets
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"input_str": "Microsoft Corporation released a new software update for Windows 11."}'
```

### Using Python requests
```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Extract triplets
data = {"input_str": "Microsoft Corporation released a new software update for Windows 11."}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### Using the Interactive Documentation
1. Navigate to http://localhost:8000/docs
2. Click on the `/predict` endpoint
3. Click "Try it out"
4. Modify the input text in the request body
5. Click "Execute" to see the results

## Configuration

### Default Input
The API includes a default input example:
```json
{
  "input_str": "Apple Inc. announced a new product launch in the technology sector."
}
```

### Model Configuration
- **Max Sequence Length**: 512 tokens
- **Device**: Automatically detects CPU/GPU availability
- **Batch Size**: 1 (for single text processing)

## Troubleshooting

### Common Issues

1. **Model weights not downloading**
   - Check internet connection
   - Verify Hugging Face access
   - Check available disk space

2. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **CUDA out of memory**
   - The model will automatically fall back to CPU
   - Reduce input text length if needed

4. **Port already in use**
   ```bash
   uvicorn src.main:app --port 8001
   ```
---