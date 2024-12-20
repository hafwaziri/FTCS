# Ticket Classification System:

During my work at Fraunhofer IPK as a Student I developed this system to automate ticket classification tasks.

This system leverages three years of historical data, comprising 20,000 tickets, to generate predictions for new tickets. It features a comprehensive Machine Learning pipeline that encompasses the entire lifecycle of model development and deployment. The pipeline includes components for data fetching, preprocessing, model training, and deployment. Additionally, it integrates an agent that performs actions based on model predictions. Also, monitoring scripts are included to oversee pipeline performance and trigger automated actions as necessary.

## System Overview
![Structure](/Report%20&%20Documentation/structure.png)

For an optimal view of the system's architecture, refer to the file:
Report & Documentation/Ticket Classification System.pdf.

Detailed documentation, technical report, including a comprehensive overview of the system's functionality, is available in:
Report & Documentation/tsdocumentation.pdf.

## Setup and Execution
The main directory contains two scripts: setup.sh and start.sh.

### setup.sh:

Designed for Debian-based Linux systems, this script:
Creates a virtual environment.
Installs all required dependencies.
For non-Debian systems or distributions, the script can be easily modified as needed.*


### start.sh:

After running setup.sh, use this script to start the server for the web interface.


##### Additional Notes:

The Agent script will be added...
A comprehensive training report and reflections on the results will also be added in the documentation...

If you're using a non-Debian OS or distribution, update the Geckodriver located in the Watchdog directory to match your specific OS/distribution.
For permission-related issues, use the following commands:
```bash
chmod +x setup.sh  
chmod +x start.sh  