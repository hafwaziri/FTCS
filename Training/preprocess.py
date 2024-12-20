import os
import json
import logging
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy
import time
from datetime import datetime

# nlp = spacy.load("de_core_news_sm")  # Load spaCy's German language model
nlp = spacy.load("de_dep_news_trf") #The version for better accuracy

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'ticket_preprocessing_script.log')
logger = logging.getLogger('ticket_preprocessing_logger')
logger.setLevel(logging.DEBUG)

# Create file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(fh)


def load_json_file(file_path: str) -> dict:
    '''Loads a JSON file and returns its content'''
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return {}


def process_resources(raw_resources_data: dict) -> dict:
    # Map Resource ID to Resource item for quick lookup
    try:
        resources_mapped = {
            item['ResID']: item for item in raw_resources_data
            if 'ResID' in item
        }
    except KeyError as e:
        logger.error(f"KeyError when mapping resources: {e}")
        return {}

    # Constructs a full resource description for a given item by recursively traversing its parent resources
    def build_full_resource_description(item) -> str:
        if item['ParentRes'] != 0:
            parent = resources_mapped.get(item['ParentRes'])
            if parent:
                parent_description = build_full_resource_description(parent)
                return f"{parent_description} \\ {item['ResBezeichnung']}"
        return item['ResBezeichnung']

    processed_resources = {}

    # Process each resource
    for item in raw_resources_data:
        try:
            res_id = item['ResID']
            full_description = build_full_resource_description(item)

            # Add to processed resources dictionary
            processed_resources[res_id] = {
                'ResourceID': res_id,
                'ResourceDescription': full_description
            }

        except KeyError as e:
            logger.error(f"KeyError when processing resource {item}: {e}")

    # Map resources to their new descriptions
    resources_to_update = {
        "REDACTED": [4, 5, 6, 7, 8, 9, 10, 11, 12, 33, 34, 44, 45, 47, 48, 49],
        "REDACTED": [28, 27, 29],
        "REDACTED": [52],
        "REDACTED": [41, 40, 39, 42, 16, 32, 38],
        "REDACTED": [24, 25, 51, 30, 15, 26, 23]
    }

    # Iterate and apply updates
    for new_value, ids_to_update in resources_to_update.items():
        for res_id in ids_to_update:
            if res_id in processed_resources:
                processed_resources[res_id]['ResourceDescription'] = new_value
            
    

    return processed_resources


def process_categories(raw_categories_data) -> dict:
    # Map Category ID to Category item for quick lookup of parent descriptions
    try:
        categories_mapped = {
            item['BetID']: item for item in raw_categories_data
            if 'BetID' in item
        }
    except KeyError as e:
        logger.error(f"KeyError when mapping categories: {e}")
        return {}

    # Constructs a full category description for a given item by recursively traversing its parent categories
    def build_full_category_description(item) -> str:
        if item['ParentID'] != 0:
            parent = categories_mapped.get(item['ParentID'])
            if parent:
                parent_description = build_full_category_description(parent)
                return f"{parent_description} \\ {item['Betreff']}"
        return item['Betreff']

    processed_categories = {}

    # Process each category
    for item in raw_categories_data:
        try:
            cat_id = item['BetID']
            full_description = build_full_category_description(item)

            # Add to processed categories dictionary
            processed_categories[cat_id] = {
                'CategoryID': cat_id,
                'CategoryDescription': full_description
            }
        except KeyError as e:
            logger.error(f"KeyError when processing category {item}: {e}")
            
    # Map categories to their new values
    categories_to_update = {
        "REDACTED": [62, 63, 171],
        "REDACTED": [137, 141, 138, 149],
        "REDACTED": [187],
        "REDACTED": [7],
        "REDACTED": [65, 66, 143, 176, 180],
        "REDACTED": [72, 73, 195, 74],
        "REDACTED": [88],
        "REDACTED": [31, 32, 90, 126, 127, 148, 150, 169, 183, 184, 192],
        "REDACTED": [92, 99],
        "REDACTED": [94, 95, 96, 98, 154, 170, 172, 190],
        "REDACTED": [100, 162, 191, 101, 102],
        "REDACTED": [76, 142, 161],
        "REDACTED": [58, 79, 103, 104, 105, 106],
        "REDACTED": [81, 128],
        "REDACTED": [112, 113],
        "REDACTED": [178, 179, 168, 46, 47, 48, 49],
        "REDACTED": [51, 167, 82, 52, 85],
        "REDACTED": [156, 107, 108]
    }

    # Iterate and apply updates
    for new_value, ids_to_update in categories_to_update.items():
        for cat_id in ids_to_update:
            if cat_id in processed_categories:
                processed_categories[cat_id]['CategoryDescription'] = new_value

    return processed_categories


def process_ticket_title(title):
    pattern = r'\[warte.*?\]|\[abholbereit.*?\]'  # regex pattern
    title = re.sub(pattern, '', title, flags=re.IGNORECASE)  # Remove patterns
    return process_text(title)  # process the text for the title before returning


def remove_html_tags(text):
    cleaned_text = BeautifulSoup(text, 'lxml').get_text()  # Use the lxml parser to parse the html and get the text
    return process_text(cleaned_text)  # Process the text and then return it


def normalize_text(token_text: str) -> str:
    """Normalizes German umlauts and returns normalized text."""
    umlaut_replacements = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}
    for umlaut, replacement in umlaut_replacements.items():
        token_text = token_text.replace(umlaut, replacement)
    return token_text


def process_text(text):
    # Use spaCy to create a Doc object from the input text
    doc = nlp(text)

    # Initialize an empty list to store processed tokens
    processed_tokens = []

    # Iterate over each token in the Doc object
    for token in doc:
        # Skip tokens that are punctuation, spaces, stop words, or shorter than 3 characters
        if token.is_punct or token.is_space or token.is_stop or len(token.text) <= 2:
            continue
        # Replace numerical tokens with a placeholder
        elif token.like_num:
            processed_tokens.append("<NUM>")
        # Replace URL or email tokens with a placeholder
        elif token.like_url or token.like_email:
            processed_tokens.append("<URL/EMAIL>")
        else:
            # Normalize the lemma of the token and convert it to lowercase
            lemma = normalize_text(token.lemma_.lower())
            processed_tokens.append(lemma)

    return processed_tokens


def process_ticket(ticket, resources_dict, categories_dict) -> dict:
    return {
        'TicketID': ticket['BunID'],  # Extract Ticket ID
        'Title': process_ticket_title(ticket['DerBetreff']),  # Extract/Process Title
        'Category': categories_dict.get(ticket['BetID'], {'CategoryDescription': 'Unknown'})['CategoryDescription'],
        'Resource': resources_dict.get(ticket['ResID'], {'ResourceDescription': 'Unknown'})['ResourceDescription'],
        'Text': remove_html_tags(ticket['Auftrag'])  # Extract/Process Ticket Description
    }


def main():
    raw_data_directory = os.path.join(script_dir, 'Data', 'Raw')

    # Load raw Ticket Data
    raw_data = load_json_file(os.path.join(raw_data_directory, 'tickets_raw.json'))
    logger.info("Raw Data loaded")

    # Load the Resources and Ticket Categories as Dictionaries
    resources_dictionary = process_resources(load_json_file(os.path.join(raw_data_directory, 'resources.json')))
    categories_dictionary = process_categories(load_json_file(os.path.join(raw_data_directory, 'categories.json')))
    logger.info("Resources and Categories raw data loaded and processed")

    if not raw_data or not resources_dictionary or not categories_dictionary:
        logger.error("Failed to load data. Exiting program.")
        return

    # Process raw tickets and append processed tickets to a list
    data = []

    # The categories and resources to be skipped. Cross Reference with categories/resources.json to know what was skipped.
    res_to_skip = [18, 19, 20, 36, 53]
    cat_to_skip = [11, 42, 43, 44, 59, 67, 68, 69, 70, 77, 78, 86, 97, 109, 110, 124, 125, 145, 146, 151, 152, 153, 157, 159, 160, 164, 185, 193]
    
    start_time = time.time()
    for ticket in tqdm(raw_data):
        if ticket['ResID'] not in res_to_skip and ticket['BetID'] not in cat_to_skip:
            data.append(process_ticket(ticket, resources_dictionary, categories_dictionary))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Raw Ticket data preprocessed in {elapsed_time} seconds")

    # Export Processed Tickets
    try:
        processed_data_path = os.path.join(script_dir, 'Data', 'Processed', 'tickets.json')
        with open(processed_data_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        logger.error(f"An error occurred while exporting processed Ticket Data: {e}")
        return
    logger.info("Cleaned Ticket data exported")


if __name__ == '__main__':
    main()