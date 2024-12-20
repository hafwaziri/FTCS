import joblib
import pandas as pd
from datetime import datetime
import json
import os
import logging

# Configure the directories/logger

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'prediction_script.log')
logger = logging.getLogger('prediction_logger')
logger.setLevel(logging.INFO)

# Create file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(fh)

def load_model(file_path: str):
    """
    Load a model from a file using joblib.

    Args:
        file_path (str): The path to the model file.

    Returns:
        The loaded model.

    Raises:
        Exception: If there is an error loading the model.
    """
    try:
        logger.info(f"Loading model from {file_path}")
        model = joblib.load(file_path)
        logger.info(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise

def predict(ticket_data: str):
    """
    Predict the category and resource for a given ticket.

    Args:
        ticket_data (str): The ticket data in JSON format.

    Returns:
        dict: The predicted category and resource.
    """
    try:
        logger.info("Starting prediction process")
        ticket = json.loads(ticket_data)
        logger.info("Ticket data loaded and parsed")

        # Load vectorizer and encoders
        vectorizer = load_model(os.path.join(script_dir, '..', '..', 'Training', 'Models', 'tfidf_vectorizer.joblib'))
        label_encoder_category = load_model(os.path.join(script_dir, '..', '..', 'Training', 'Models', 'label_encoder_category.joblib'))
        label_encoder_resource = load_model(os.path.join(script_dir, '..', '..', 'Training', 'Models', 'label_encoder_resource.joblib'))

        # Load the models
        category_model = load_model(os.path.join(script_dir, '..', '..', 'Training', 'Models', 'model_category.joblib'))
        resource_model = load_model(os.path.join(script_dir, '..', '..', 'Training', 'Models', 'model_resource.joblib'))

        # Prepare single ticket data for prediction
        df_single = pd.DataFrame([ticket])
        df_single['combined_text'] = df_single['Title'].apply(' '.join) + df_single['Text'].apply(' '.join)
        logger.info("Ticket data prepared for prediction")

        # Transform text data using TF-IDF vectorizer
        X_single_vector = vectorizer.transform(df_single['combined_text'])
        logger.info("Text data transformed using TF-IDF vectorizer")

        # Predict category and resource
        y_category_pred = category_model.predict(X_single_vector)
        y_resource_pred = resource_model.predict(X_single_vector)
        logger.info("Predictions made for category and resource")

        # Inverse transform predictions to original labels
        predicted_category = label_encoder_category.inverse_transform(y_category_pred)
        predicted_resource = label_encoder_resource.inverse_transform(y_resource_pred)
        logger.info("Predictions inverse transformed to original labels")

        # Prepare response
        response = {
            "Predicted Category": predicted_category[0],
            "Predicted Resource": predicted_resource[0]
        }
        logger.info("Prediction process completed successfully")

        return response
    except Exception as e:
        logger.error(f"Error during prediction process: {e}")
        raise