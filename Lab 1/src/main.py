from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import List
from .predict import predict_triplets
from .load_weights import download_weights

download_weights()
app = FastAPI(
    title="BERT BIO Tagging API",
    description="A machine learning API for extracting subject-predicate-object triplets from text using BERT-based BIO tagging",
)

class ModelData(BaseModel):
    """Input data model for text prediction"""
    input_str: str = "Apple Inc. announced a new product launch in the technology sector."

    
class Triplet(BaseModel):
    """Triplet model representing subject-predicate-object relationship"""
    subject: str = "Apple Inc."
    predicate: str = "announced"
    object: str = "product launch"

class ModelResponse(BaseModel):
    """Response model containing list of extracted triplets"""
    response: List[Triplet]

@app.get("/", 
         status_code=status.HTTP_200_OK,
         summary="Health Check",
         description="Check if the API service is running and healthy",
         response_description="Returns the health status of the API")
async def health_ping():
    """
    Health check endpoint to verify API availability.
    
    Returns:
        dict: Status indicating API health
    """
    return {"status": "healthy"}

@app.post("/predict", 
          response_model=ModelResponse,
          summary="Extract Triplets from Text",
          description="Extract subject-predicate-object triplets from input text using BERT-based BIO tagging",
          response_description="List of extracted triplets with subject, predicate, and object components")
async def predict(data: ModelData):
    """
    Extract subject-predicate-object triplets from input text.
    
    This endpoint uses a pre-trained BERT model with BIO tagging to identify and extract
    structured relationships from unstructured text. The model identifies subjects, predicates,
    and objects within the text and forms meaningful triplets.
    
    Args:
        data (ModelData): Input containing the text string to process
        
    Returns:
        ModelResponse: List of extracted triplets
        
    Raises:
        HTTPException: 500 error if prediction fails
    """
    try:
        prediction = predict_triplets(data.input_str)
        triplets = [Triplet(subject=s, predicate=p, object=o) for s, p, o in prediction]
        return ModelResponse(response=triplets)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
