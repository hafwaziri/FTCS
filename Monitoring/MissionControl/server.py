from flask import Flask, render_template, send_file, request, jsonify, abort
import os
import sys
import psutil
from datetime import datetime
import subprocess
import threading
import logging
import sqlite3

# Set up the pipeline path and import the pipeline module
# Current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.abspath(os.path.join(script_dir, '../../Deployment/Pipeline'))
sys.path.append(pipeline_path)
import pipeline

#use the python from the virtual environment
venv_python = '../../tsvenv/bin/python'

# Initialize the Flask application
app = Flask(__name__)

# Configure the directories/logger

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'server_script.log')
logger = logging.getLogger('server_logger')
logger.setLevel(logging.INFO)

# Create file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

@app.route('/')
def index():
    """
    Route for the home page.
    Renders the index.html template.
    
    Returns:
        str: Rendered HTML content.
    """
    return render_template('index.html')

def get_log_dates():
    """
    Get all date folders from the logs directory.
    
    Returns:
        list: A list of dates sorted in reverse order.
    """
    logs_dir = os.path.join(script_dir, '..', 'logs')
    if os.path.exists(logs_dir):
        return sorted(os.listdir(logs_dir), reverse=True)
    return []

def get_files_for_date(date):
    """
    Get all files for a specific date.
    
    Args:
        date (str): The date folder name.
    
    Returns:
        list: A list of filenames sorted in order.
    """
    logs_dir = os.path.join(script_dir, '..', 'logs')
    date_path = os.path.join(logs_dir, date)
    if os.path.exists(date_path):
        return sorted(os.listdir(date_path))
    return []

@app.route('/logs')
def logs():
    """
    Route for the logs page.
    Retrieves all log dates and their corresponding files.
    Renders the logs.html template with the logs data.
    
    Returns:
        str: Rendered HTML content.
    """
    dates = get_log_dates()
    logs_data = {date: get_files_for_date(date) for date in dates}
    return render_template('logs.html', logs_data=logs_data)

@app.route('/logs/<date>/<filename>')
def serve_file(date, filename):
    """
    Serve the requested log file.
    
    Args:
        date (str): The date folder name.
        filename (str): The name of the file to serve.
    
    Returns:
        Response: The file content or an error message.
    """
    logs_dir = os.path.join(script_dir, '..', 'logs')
    file_path = os.path.join(logs_dir, date, filename)
    if filename.endswith('.png'):
        return send_file(file_path, mimetype='image/png')
    else:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        except Exception as e:
            return str(e), 500

@app.route('/predict/<ticket>', methods=['GET'])
def predict(ticket):
    """
    Predict the outcome for a given ticket using the pipeline module.
    
    Args:
        ticket (str): The ticket identifier.
    
    Returns:
        Response: JSON response containing the prediction result.
    """
    if not ticket:
        abort(400, description="Invalid ticket")
    try:
        result = pipeline.process_ticket(ticket)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        abort(500, description="Prediction failed")

@app.route('/data')
def data():
    """
    Fetch all records from the 'tickets' table in the database and return them as JSON.
    
    Returns:
        Response: JSON response containing all ticket records.
    """
    conn = get_db_connection()
    if conn is None:
        abort(500, description="Database connection failed")
    try:
        tickets = conn.execute('SELECT * FROM tickets').fetchall()
        return jsonify([dict(row) for row in tickets])
    except sqlite3.Error as e:
        logging.error(f"Database query error: {e}")
        abort(500, description="Database query failed")
    finally:
        conn.close()

def get_db_connection():
    """
    Establish a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: SQLite connection object if successful, None otherwise.
    """
    try:
        db_path = os.path.join(pipeline_path, '..', 'Data', 'tickets_db.db')
        conn = sqlite3.connect(os.getenv('DB_PATH', db_path))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        return None

@app.route('/predictions')
def predictions():
    """
    Render the predictions page.
    
    Returns:
        str: Rendered HTML content.
    """
    return render_template('predictions.html')

def run_training_script(data_op, training_type, training_args):
    """
    Run the training script with the specified arguments.
    
    Args:
        data_op (str): Data operation argument.
        training_type (str): Training type argument.
        training_args (str): Additional training arguments.
    
    Returns:
        dict: Dictionary containing the success status and output of the script.
    """
    try:
        script_path = os.path.abspath(os.path.join(script_dir, '../../Training/training_pipeline.py'))
        cmd = [venv_python, script_path, data_op, training_type, training_args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/train')
def train_page():
    """
    Render the training page.
    
    Returns:
        str: Rendered HTML content.
    """
    return render_template('train.html')

@app.route('/api/train', methods=['POST'])
def trigger_training():
    """
    Trigger the training process.
    Runs the training script in a separate thread to avoid blocking the response.
    
    Returns:
        Response: JSON response indicating the training process has started.
    """
    data = request.json
    data_op = data.get('data_operation', 'd')
    training_type = data.get('training_type', 'd')
    training_args = data.get('training_args', '')

    def run_training():
        run_training_script(data_op, training_type, training_args)

    thread = threading.Thread(target=run_training)
    thread.start()

    return jsonify({
        'status': 'started',
        'message': 'Training process has been initiated'
    })

PROCESSES = {
    'watchdog': None,
    'pipeline': None,
    'action-agent': None
}

def get_script_status(script_name):
    """
    Check if a script is running and get its details.
    
    Args:
        script_name (str): The name of the script.
    
    Returns:
        dict: Dictionary containing the status, PID, and running time of the script.
    """
    process = PROCESSES.get(script_name)
    if process is None:
        return {'status': 'stopped', 'pid': None, 'running_time': None}
    try:
        if process.poll() is None:
            p = psutil.Process(process.pid)
            start_time = datetime.fromtimestamp(p.create_time())
            running_time = str(datetime.now() - start_time).split('.')[0]
            return {
                'status': 'running',
                'pid': process.pid,
                'running_time': running_time
            }
        else:
            PROCESSES[script_name] = None
            return {'status': 'stopped', 'pid': None, 'running_time': None}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        PROCESSES[script_name] = None
        return {'status': 'stopped', 'pid': None, 'running_time': None}

def start_script(script_name):
    """
    Start a script if it's not already running.
    
    Args:
        script_name (str): The name of the script to start.
    
    Returns:
        tuple: A tuple containing a boolean indicating success and a message.
    """
    if PROCESSES.get(script_name) is not None:
        return False, "Script is already running"

    script_paths = {
        'watchdog': os.path.join(script_dir, '..', '..', 'Deployment', 'WatchDog', 'watchdog.py'),
        'pipeline': os.path.join(script_dir, '..', '..', 'Deployment', 'Pipeline', 'pipeline.py'),
        'action-agent': os.path.join(script_dir, '..', '..', 'Deployment', 'Agent', 'actionagent.py')
    }

    if script_name not in script_paths:
        return False, f"Invalid script name: {script_name}"

    script_path = script_paths[script_name]

    try:
        process = subprocess.Popen(
            [venv_python, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        PROCESSES[script_name] = process
        return True, f"Started {script_name}"
    except Exception as e:
        return False, str(e)

def stop_script(script_name):
    """
    Stop a running script.
    
    Args:
        script_name (str): The name of the script to stop.
    
    Returns:
        tuple: A tuple containing a boolean indicating success and a message.
    """
    process = PROCESSES.get(script_name)
    if process is None:
        return False, "Script is not running"
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        PROCESSES[script_name] = None
        return True, f"Stopped {script_name}"
    except Exception as e:
        return False, str(e)

@app.route('/monitor')
def monitor_page():
    """
    Render the monitor page.
    
    Returns:
        str: Rendered HTML content.
    """
    return render_template('monitor.html')

@app.route('/api/status')
def get_status():
    """
    Get the status of the scripts.
    
    Returns:
        Response: JSON response containing the status of the watchdog and pipeline scripts.
    """
    return jsonify({
        'watchdog': get_script_status('watchdog'),
        'pipeline': get_script_status('pipeline'),
        'action-agent': get_script_status('action-agent')
    })

@app.route('/api/control/<script_name>/<action>', methods=['POST'])
def control_script(script_name, action):
    """
    Control the start/stop actions for scripts.
    
    Args:
        script_name (str): The name of the script.
        action (str): The action to perform ('start' or 'stop').
    
    Returns:
        Response: JSON response indicating the success of the action and the current status of the script.
    """
    if action == 'start':
        success, message = start_script(script_name)
    elif action == 'stop':
        success, message = stop_script(script_name)
    else:
        return jsonify({'error': 'Invalid action'}), 400

    return jsonify({
        'success': success,
        'message': message,
        'status': get_script_status(script_name)
    })

if __name__ == '__main__':
    # Run the Flask application in debug mode on all available IP addresses at port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)