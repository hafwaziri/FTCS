import sqlite3
import threading
import time
from tsmessenger import get_ticket_details
from ticketprocessing import preprocess_ticket
from predictionmodel import predict
from datetime import datetime
import json
import logging
import os

# Configure the directories/logger

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'pipeline_script.log')
logger = logging.getLogger('pipeline_logger')
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

# Function to process a ticket given its ID
def process_ticket(ticket_id):
    """
    Process a ticket by fetching its details, preprocessing the data, and predicting the outcome.

    Args:
        ticket_id (int): The ID of the ticket to process.

    Returns:
        str: The prediction result as a JSON string, or None if an error occurs.
    """
    try:
        raw_ticket_data = get_ticket_details(ticket_id)
        ticket_data = preprocess_ticket(raw_ticket_data)
        response = predict(ticket_data)
        return json.dumps(response)
    except Exception as e:
        logger.error(f"Error processing ticket ID {ticket_id}: {e}")
        return None

# Function to monitor the database for unprocessed tickets
def monitor_db(db_path, sleep_interval=10):
    """
    Monitor the database for unprocessed tickets and process them.

    Args:
        db_path (str): The path to the SQLite database.
        sleep_interval (int): The interval in seconds to wait between checks.
    """
    while True:
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, ticket_id FROM tickets WHERE Processed = 0")
                tickets = cursor.fetchall()

                for ticket in tickets:
                    ticket_db_id = ticket[0]
                    ticket_id = ticket[1]
                    prediction = process_ticket(ticket_id)
                    if prediction:
                        cursor.execute("""
                            UPDATE tickets
                            SET Processed = 1, Prediction = ?
                            WHERE id = ?
                        """, (prediction, ticket_db_id))
                        conn.commit()
                        logger.info(f"Processed Ticket ID: {ticket_id}, Prediction: {prediction}")

            logger.info('Monitoring Database...')
            time.sleep(sleep_interval)

        except sqlite3.DatabaseError as db_err:
            logger.error(f"Database error: {db_err}")
        except Exception as e:
            logger.error(f"Error in monitoring database: {e}")

if __name__ == '__main__':
    db_path = os.path.join(script_dir, '..', 'Data', 'tickets_db.db')
    monitor_thread = threading.Thread(target=monitor_db, args=(db_path,))
    monitor_thread.daemon = True
    monitor_thread.start()

    while True:
        time.sleep(1)