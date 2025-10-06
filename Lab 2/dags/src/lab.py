import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import pickle
import os
import json

def load_data():
    """
    Loads the iris dataset from sklearn and saves it to a shared file.
    Returns:
        str: Path to the saved data file.
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)

    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target  # Add target column for reference
    
    # Save to CSV file for easy access
    data_file = os.path.join(data_dir, "iris_data.csv")
    df.to_csv(data_file, index=False)
    
    # Also save metadata as JSON
    metadata = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'n_samples': len(df),
        'n_features': len(iris.feature_names)
    }
    metadata_file = os.path.join(data_dir, "iris_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    return data_file

def data_preprocessing(data_file_path: str):
    """
    Reads data from file, performs preprocessing, and saves preprocessed data to file.
    Returns:
        str: Path to the preprocessed data file.
    """
    # Read data from CSV file
    df = pd.read_csv(data_file_path)
    
    # Load metadata
    metadata_file = os.path.join(os.path.dirname(data_file_path), "iris_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    df = df.dropna()
    
    # Separate features and target for classification
    feature_columns = metadata['feature_names']
    X = df[feature_columns]
    y = df['target']
    
    # Apply MinMax scaling to features
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    
    # Create preprocessed data directory
    preprocessed_dir = os.path.join(os.path.dirname(data_file_path), "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Save preprocessed features and target
    preprocessed_file = os.path.join(preprocessed_dir, "iris_preprocessed.pkl")
    preprocessed_data = {
        'X': X_scaled,
        'y': y.values,
        'feature_names': feature_columns,
        'scaler': min_max_scaler,
        'target_names': metadata['target_names']
    }
    
    with open(preprocessed_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    return preprocessed_file


def build_save_model(preprocessed_file_path: str, filename: str):
    """
    Builds a Logistic Regression model on the preprocessed iris data and saves it.
    Returns the model performance metrics (JSON-serializable).
    """
    # Load preprocessed data from file
    with open(preprocessed_file_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    scaler = preprocessed_data['scaler']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Logistic Regression model
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = log_reg.predict(X_test)
    y_pred_proba = log_reg.predict_proba(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save the model and scaler
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    model_data = {
        'model': log_reg,
        'scaler': scaler,
        'feature_names': preprocessed_data['feature_names'],
        'target_names': preprocessed_data['target_names'],
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    # Save metrics to JSON file for easy access
    metrics_file = os.path.join(output_dir, "model_metrics.json")
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Return performance metrics for XCom
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }


def load_model_elbow(filename: str, metrics: dict):
    """
    Loads the saved Logistic Regression model and evaluates it on new test data.
    Returns the prediction for the first test sample and model evaluation summary.
    """
    # Load the saved model data
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model_data = pickle.load(open(output_path, "rb"))
    
    loaded_model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    target_names = model_data['target_names']
    
    # Create test data from iris dataset (using a subset for testing)
    iris = load_iris()
    test_data = iris.data[:5]  # Use first 5 samples as test data
    test_targets = iris.target[:5]  # True labels for comparison
    
    # Apply the same preprocessing as training data
    test_data_scaled = scaler.transform(test_data)
    
    # Make predictions
    predictions = loaded_model.predict(test_data_scaled)
    prediction_proba = loaded_model.predict_proba(test_data_scaled)
    
    # Calculate accuracy on this small test set
    test_accuracy = accuracy_score(test_targets, predictions)
    
    # Get class names for better interpretation
    class_names = target_names
    
    # Create detailed results for each sample
    sample_results = []
    for i in range(len(predictions)):
        true_class = class_names[test_targets[i]]
        pred_class = class_names[predictions[i]]
        confidence = max(prediction_proba[i])
        sample_results.append({
            'sample': i+1,
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': float(confidence)
        })
    
    # Save evaluation results to file
    eval_dir = os.path.join(os.path.dirname(output_path), "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    eval_file = os.path.join(eval_dir, "evaluation_results.json")
    
    evaluation_results = {
        'test_accuracy': test_accuracy,
        'training_accuracy': model_data['accuracy'],
        'sample_results': sample_results,
        'predictions': predictions.tolist(),
        'test_targets': test_targets.tolist()
    }
    
    with open(eval_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Return the first prediction and evaluation summary
    result = {
        'first_prediction': int(predictions[0]),
        'first_prediction_class': class_names[predictions[0]],
        'test_accuracy': test_accuracy,
        'all_predictions': predictions.tolist(),
        'training_accuracy': model_data['accuracy'],
        'evaluation_file': eval_file
    }
    
    return result
