import json
import os
import logging
from decimal import Decimal
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'datafetch_script.log')
logger = logging.getLogger('datafetch_logger')
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

class DecimalEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for Decimal and datetime objects.
    Converts Decimal to float and datetime to ISO format string.
    """
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)

def fetch_and_export_data(cursor, query, output_file):
    """
    Fetch data from the database using the provided cursor and query,
    then export the data to a JSON file.

    Args:
        cursor (pyodbc.Cursor): The database cursor to execute the query.
        query (str): The SQL query to fetch data.
        output_file (str): The path to the output JSON file.
    """
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]
        dump_file = os.path.join(script_dir, 'Data', 'Raw', output_file)
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        with open(dump_file, 'w') as json_file:
            json.dump(data, json_file, indent=4, cls=DecimalEncoder)
        logger.info(f"Data has been exported to {output_file}")
    except Exception as e:
        logger.error(f"Error fetching or exporting data: {e}")

def main():
    """
    Main function to establish a database connection and export data
    from specific tables to JSON files.
    """
    try:
        # Construct the connection string using environment variables
        connection_string = (
            'DRIVER={FreeTDS};SERVER=REDACTED;PORT=REDACTED;DATABASE=REDACTED;UID=REDACTED;PWD=REDACTED;TDS_Version=8.0;'
        )
        with pyodbc.connect(connection_string) as connection:
            logger.info("Connection to SQL Server successful")
            with connection.cursor() as cursor:
                # Fetch and export data from specified tables
                fetch_and_export_data(cursor, "REDACTED", 'resources.json')
                fetch_and_export_data(cursor, "REDACTED", 'categories.json')
                fetch_and_export_data(cursor, "REDACTED", 'tickets_raw.json')
    except pyodbc.Error as e:
        logger.error(f"Error connecting to SQL Server: {e}")

if __name__ == "__main__":
    main()