import requests
import json
import os
import logging
from datetime import datetime

# Configure the directories/logger

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'ticketsystem_messenger_script.log')
logger = logging.getLogger('ticketsystem_messenger_logger')
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

# Bearer token for authentication from environment variable
bearer_token = os.getenv('BEARER_TOKEN')

def make_api_request(url, headers):
    """Helper function to make API requests and handle errors."""
    logger.info(f"Making API request to URL: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()
    logger.info(f"API request to URL: {url} successful")
    return response

def get_ticket_details(ticket):
    """
    Get the details of a ticket including its title and specific step text.

    Args:
        ticket (int): The ticket ID.

    Returns:
        str: JSON formatted string containing the ticket title and step text.
    """
    logger.info(f"Fetching details for ticket ID: {ticket}")

    # Headers for the API request
    headers = {
        'Authorization': f'Bearer {bearer_token}'
    }

    # Request to get the ticket details
    title_response = make_api_request(f'http://redacted/{ticket}', headers)
    title = json.loads(title_response.text)
    logger.debug(f"Title: {title['fields']['subject']}")

    # Request to get the steps of the ticket
    id_response = make_api_request(f'http://redacted/{ticket}/steps', headers)
    id_data = json.loads(id_response.text)

    # Find the step with actionID 4 to get the step ID
    step_id = next((entry['id'] for entry in id_data if entry['actionID'] == 4), None)
    if step_id is None:
        logger.error("Step with actionID 4 not found.")
        return json.dumps({'error': 'Step with actionID 4 not found'}, indent=4)

    # Request to get the text of the specific step
    text_response = make_api_request(f'http://redacted/{ticket}/steps/{step_id}', headers)
    text = json.loads(text_response.text)
    logger.debug(f"Text: {text['text']}")

    # Prepare the data to be returned
    data = {
        'Title': title['fields']['subject'],
        'Text': text['text'],
    }

    # Convert the data to JSON format
    json_data = json.dumps(data, indent=4)

    logger.info(f"Details for ticket ID: {ticket} fetched successfully")

    # Return the JSON data
    return json_data