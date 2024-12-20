import sys
import os
import preprocess
import datafetch
import cputraining
import gputraining
import logging
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize logger and configure the directories/logger

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'training_pipeline_script.log')
logger = logging.getLogger('training_pipeline_logger')
logger.setLevel(logging.INFO)

# Create file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)

def main():
    """
    Main function to control the training pipeline based on command line arguments.
    
    Command line arguments:
    sys.argv[1]: Specifies the operation to perform.
        'df' - Fetch data and preprocess it.
        'pp' - Only preprocess the data.
        'd'  - Do nothing.
    sys.argv[2]: Specifies the training type.
        'cpu' - Perform CPU training.
        'gpu' - Perform GPU training.
        'd'   - Do nothing.
    sys.argv[3]: Additional argument passed to the training function.
    """
    
    logger.info('Starting the training pipeline script')
    
    if len(sys.argv) < 4:
        logger.error('Not enough command line arguments provided')
        sys.exit(1)
    
    operation = sys.argv[1]
    training_type = sys.argv[2]
    additional_arg = sys.argv[3]
    
    logger.info(f'Operation: {operation}, Training Type: {training_type}, Additional Arg: {additional_arg}')
    
    if operation == 'df':
        logger.info('Fetching data and preprocessing it')
        datafetch.main()
        preprocess.main()
    elif operation == 'pp':
        logger.info('Only preprocessing the data')
        preprocess.main()
    elif operation == 'd':
        logger.info('Doing nothing for data operation')
    
    if training_type == 'cpu':
        logger.info('Performing CPU training')
        cputraining.main(additional_arg)
    elif training_type == 'gpu':
        logger.info('Performing GPU training')
        gputraining.main(additional_arg)
    elif training_type == 'd':
        logger.info('Doing nothing for training operation')
    
    logger.info('Training pipeline script finished')

if __name__ == '__main__':
    main()