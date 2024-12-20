import json
from bs4 import BeautifulSoup
import re
import spacy
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
log_file_path = os.path.join(date_dir, 'ticketprocessing_script.log')
logger = logging.getLogger('ticketprocessing_logger')
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

# nlp = spacy.load("de_core_news_sm") # Load spaCy's German language model (small version for efficiency)
nlp = spacy.load("de_dep_news_trf") #Large version for accuracy


def remove_html_tags(text):
    cleaned_text = BeautifulSoup(text, 'lxml').get_text() # Use the lxml parser to parse the html and get the text
    return process_text(cleaned_text) # Process the text and then return it

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
    
def process_ticket_title(title):
    pattern = r'\[warte.*?\]|\[abholbereit.*?\]'  # regex pattern
    title = re.sub(pattern, '', title, flags=re.IGNORECASE)  # Remove patterns
    return process_text(title) # Process the text for the title before returning

def preprocess_ticket(json_data):
    try:
        data = json.loads(json_data)
        logger.info(f"Data received by ticketprocessing: {data}")
        
        data['Title'] = process_ticket_title(data['Title'])
        data['Text'] = remove_html_tags(data['Text'])
        
        logger.info(f"Processed data: {data}")
        
        return json.dumps(data, indent=4)
    except Exception as e:
        logger.error(f"Error processing ticket: {e}")
        return None