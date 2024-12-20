#!/bin/bash

# Exit on any error
set -e

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# Check for pip
if ! command -v pip3 &> /dev/null
then
    echo "pip3 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

sudo apt-get install -y python3-venv

# Create virtual environment
echo "Creating virtual environment 'tsvenv'..."
python3 -m venv tsvenv

# Activate virtual environment
source tsvenv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install \
    datetime \
    logging \
    joblib \
    pandas \
    beautifulsoup4 \
    lxml \
    spacy \
    requests \
    selenium \
    flask \
    psutil \
    scikit-learn \
    scikit-learn-intelex \
    matplotlib \
    numpy \
    pyodbc \
    tqdm 


# Additional language model download for spacy
python3 -m spacy download de_core_news_sm
python3 -m spacy download de_dep_news_trf



#Firefox DEB Install for Watchdog
sudo snap remove firefox
sudo install -d -m 0755 /etc/apt/keyrings
wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | sudo tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | sudo tee -a /etc/apt/sources.list.d/mozilla.list > /dev/null
echo '
Package: *
Pin: origin packages.mozilla.org
Pin-Priority: 1000

Package: firefox*
Pin: release o=Ubuntu
Pin-Priority: -1' | sudo tee /etc/apt/preferences.d/mozilla
sudo apt update && sudo apt remove firefox
sudo apt install firefox

chmod +x Deployment/WatchDog/geckodriver


echo "Environment setup complete."
echo "Press any key to continue..."
read -n 1 -s
