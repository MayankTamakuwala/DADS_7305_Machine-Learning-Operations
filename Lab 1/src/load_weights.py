import os
import requests

def download_weights():
    """
    Download pre-trained BERT model weights from Hugging Face if not already present.
    
    This function checks if the model weights file exists locally, and if not,
    downloads it from the Hugging Face model repository. The weights are essential
    for the BERT model to perform BIO tagging predictions.
    """
    # Check if model weights file already exists locally
    if not os.path.exists("./model/BERT_BIO_Tagging_model.pth"):
        # URL to the pre-trained model weights on Hugging Face
        url = "https://huggingface.co/MayankTamakuwala/BERT_BIO_Tagger/resolve/main/BERT_BIO_Tagging_model.pth"
        output_path = "./model/BERT_BIO_Tagging_model.pth"

        # Download the model weights with streaming to handle large files efficiently
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Write the downloaded weights to local file in chunks to manage memory
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                f.write(chunk)

        # Log successful download for monitoring and debugging
        print(f"Model weights downloaded to {output_path}")
    # else:
    #     print(f"Weights already exists")
