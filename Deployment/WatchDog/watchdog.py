import os
import logging
import sqlite3
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException
from datetime import datetime

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(script_dir, '../Data/tickets_db.db')  # Path to the SQLite database
GECKODRIVER_PATH = os.path.join(script_dir, './geckodriver')  # Path to the GeckoDriver executable
TICKET_SYSTEM_URL = os.getenv('TICKET_SYSTEM_URL')  # URL of the ticket system
USERNAME = os.getenv('TICKET_SYSTEM_USERNAME')  # Username for the ticket system
PASSWORD = os.getenv('TICKET_SYSTEM_PASSWORD')  # Password for the ticket system

# Configure the directories/logger

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'watchdog_script.log')
logger = logging.getLogger('ticketsystem_watchdog_logger')
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

def init_db():
    """
    Initialize the SQLite database and create the tickets table if it doesn't exist.
    """
    conn = sqlite3.connect(DATABASE_PATH)  # Connect to the SQLite database
    cursor = conn.cursor()  # Create a cursor object
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY,
        ticket_id INTEGER UNIQUE,
        Prediction TEXT,
        Processed INTEGER,
        Action INTEGER
    )
    ''')  # Create the tickets table if it doesn't exist
    return conn, cursor  # Return the connection and cursor

def save_ticket(cursor, conn, ticket_id):
    """
    Save a new ticket to the database.
    """
    try:
        cursor.execute('INSERT INTO tickets (ticket_id, Processed, Action) VALUES (?,?,?)', (ticket_id, 0, 0))  # Insert a new ticket
        conn.commit()  # Commit the transaction
        logger.info("TICKET SAVED")  # Log that the ticket was saved
    except sqlite3.IntegrityError:
        logger.warning(f"Ticket ID {ticket_id} already exists in the database.")  # Log a warning if the ticket ID already exists

def login(driver, wait):
    """
    Log in to the ticket system using Selenium.
    """
    try:
        driver.get(TICKET_SYSTEM_URL)  # Open the ticket system URL
        einloggen_button = wait.until(EC.element_to_be_clickable((By.XPATH, REDACTED)))
        einloggen_button.click()  # Click the login button

        benutzername_button = wait.until(EC.element_to_be_clickable((By.XPATH, REDACTED)))  # Wait for the username button to be clickable
        benutzername_button.click()  # Click the username button

        username_input = wait.until(EC.presence_of_element_located((By.ID, "inputUsername")))  # Wait for the username input to be present
        username_input.send_keys(USERNAME)  # Enter the username

        password_input = wait.until(EC.presence_of_element_located((By.ID, "inputPassword")))  # Wait for the password input to be present
        password_input.send_keys(PASSWORD)  # Enter the password

        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "REDACTED")))  # Wait for the login button to be clickable
        login_button.click()  # Click the login button
    except TimeoutException as e:
        logger.error(f"Timeout during login: {e}")  # Log an error if there is a timeout
    except WebDriverException as e:
        logger.error(f"WebDriver error during login: {e}")  # Log an error if there is a WebDriver exception

def process_tickets(driver, wait, cursor, conn):
    """
    Process new tickets by checking the ticket system periodically.
    """
    tickets_seen = set()  # Set to keep track of seen tickets
    last_refresh_time = time.time()  # Track the last refresh time
    while True:
        try:
            if time.time() - last_refresh_time > 3600:  # Refresh every hour
                driver.refresh()  # Refresh the page
                last_refresh_time = time.time()  # Update the last refresh time
                logger.info("Page refreshed")  # Log that the page was refreshed

            ticket_elements = wait.until(EC.presence_of_all_elements_located((By.XPATH, REDACTED)))  # Wait for ticket elements to be present
            for ticket_element in ticket_elements:
                ticket_id = ticket_element.get_attribute("ticketid")  # Get the ticket ID
                if ticket_id not in tickets_seen:  # If the ticket ID is new
                    logger.info(f"New Ticket ID: {ticket_id}")  # Log the new ticket ID
                    tickets_seen.add(ticket_id)  # Add the ticket ID to the set of seen tickets
                    save_ticket(cursor, conn, ticket_id)  # Save the ticket to the database
            time.sleep(60)  # Wait for a minute before checking again
        except TimeoutException as e:
            logger.error(f"Timeout while looking for tickets: {e}")  # Log an error if there is a timeout
            time.sleep(5)  # Wait for 5 seconds before retrying
        except WebDriverException as e:
            logger.error(f"WebDriver error while looking for tickets: {e}")  # Log an error if there is a WebDriver exception
            time.sleep(5)  # Wait for 5 seconds before retrying
        except Exception as e:
            logger.error(f"Unexpected error: {e}")  # Log any unexpected errors
            time.sleep(5)  # Wait for 5 seconds before retrying
            driver.quit()  # Quit the driver
            main()  # Restart from the beginning

def main():
    """
    Main function to initialize the database, start the WebDriver, and process tickets.
    """
    conn, cursor = init_db()
    service = Service(GECKODRIVER_PATH)
    with webdriver.Firefox(service=service) as driver:  # Start the Firefox WebDriver
        wait = WebDriverWait(driver, 10)  # Create a WebDriverWait object
        try:
            login(driver, wait)
            time.sleep(5)
            tickets_module = wait.until(EC.element_to_be_clickable((By.XPATH, REDACTED)))  # Wait for the tickets module to be clickable
            tickets_module.click()
            time.sleep(2)
            driver.get(REDACTED)
            process_tickets(driver, wait, cursor, conn)
        finally:
            conn.close()  # Close the database connection

if __name__ == "__main__":
    main()