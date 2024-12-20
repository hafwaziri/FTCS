from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearnex import patch_sklearn
from datetime import datetime
from joblib import dump
import pandas as pd
import json
import matplotlib.pyplot as plt
import logging
import numpy as np
import sys
import os
import time

#Scikit-learn optimization for Intel CPUs:
patch_sklearn()

script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure the directories/logger

# Get current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Create directories for logging with current date/time
date_dir = os.path.join(script_dir, '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

# Set up logging to use the date_dir
log_file_path = os.path.join(date_dir, 'cputraining_script.log')
logger = logging.getLogger('cputraining_logger')
logger.setLevel(logging.INFO)

# Create file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
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
        logger.error(f"An unexpected error occured: {e}")
    return {}

def input_report(df: object, type: str):
    count = df[type].value_counts()
    
    #Generate and save plots to the disk
    plt.figure(figsize=(16, 20))
    ax = count.plot(kind='barh')
    plt.xlabel('Count')
    plt.ylabel(f'{type}')
    plt.title(f'{type} Counts')

    # Add the number to the top of each bar
    for index, value in enumerate(count):
        plt.text(value, index, str(value))

    plt.savefig(os.path.join(date_dir, f'{type}_counts_plot.png'), dpi=300, bbox_inches='tight')

def generate_report(model, classification_report, elapsed_time):

    #Write the reports of the models to the disk
    
    with open(os.path.join(date_dir, f'{str(model)}_report.txt'), 'a') as file:
        file.write("#" * 200 + "\n")
        file.write("Report generated at: " + datetime.now().strftime('%H-%M-%S') + "\n")
        file.write(classification_report + '\n')
        file.write("Elapsed Time: " +  str(elapsed_time) + ' seconds' + "\n")

def save_model(model, model_name, model_type):
    """
    Save a trained model to the Models directory
    
    Parameters:
    model: trained sklearn model
    model_name: str, name of the algorithm (lr, svm, rf, gbm)
    model_type: str, type of prediction (categories or resources)
    """
    
    # Create Models directory if it doesn't exist
    models_dir = os.path.join(script_dir, 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate filename based on model name and type
    if model_name == 'best':
        filename = f'model_{model_type}.joblib'
    else:
        filename = f'{model_name}_{model_type}.joblib'
    
    # Save the model
    model_path = os.path.join(models_dir, filename)
    dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def save_preprocessors(vectorizer, label_encoder_category, label_encoder_resource):
    """
    Save the TF-IDF vectorizer and label encoders to the Models directory
    
    Parameters:
    vectorizer: fitted TfidfVectorizer
    label_encoder_category: fitted LabelEncoder for categories
    label_encoder_resource: fitted LabelEncoder for resources
    """
        
    # Create Models directory if it doesn't exist
    models_dir = os.path.join(script_dir, 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the vectorizer
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    dump(vectorizer, vectorizer_path)
    logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")
    
    # Save the label encoders
    category_encoder_path = os.path.join(models_dir, 'label_encoder_category.joblib')
    resource_encoder_path = os.path.join(models_dir, 'label_encoder_resource.joblib')
    dump(label_encoder_category, category_encoder_path)
    dump(label_encoder_resource, resource_encoder_path)
    logger.info("Label encoders saved")

def tfidf_vectorizer(train_set, test_set):
    
    #noisy terms
    custom_stopwords = ['misc', 'org', 'loc', 'der', 'zu', 'und', 'sein', 'in', 'auf', 'per','an', 'für', 'ich', 'ipk','grüß','von','mit', 'können', 'nach', 'werden']

    
    # Initialize the TfidfVectorizer with specific parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,          # Limit the number of features to 5000
        ngram_range=(1,2),          # Consider unigrams and bigrams
        max_df=0.8,                 # Ignore terms that appear in more than 80% of the documents
        min_df=0.01,                # Ignore terms that appear in less than 1% of the documents
        smooth_idf=True,            # Smooth idf weights by adding one to document frequencies
        sublinear_tf=True,          # Apply sublinear tf scaling
        norm='l2',                  # Apply L2 normalization to the output vectors
        stop_words=custom_stopwords # Use custom stop words
    )
    X_train_vec = vectorizer.fit_transform(train_set) #Fit and Transform the training set
    X_test_vec = vectorizer.transform(test_set) #Transform the test set

    return vectorizer, X_train_vec, X_test_vec

def logistic_regression(X_train, X_test, y_train, y_test, label_encoder, model_type):
    
    logger.info("Fitting the Logistic Regression Model")
    
    # Record the start time
    start_time = time.time()
    
    #Initialize the Logistic Regression Model
    model = LogisticRegression()
    
    #Create a Stratified K-Fold Cross Validator
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    #Define the Parameter Grid for Hyperparameter tuning
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', None], #Regularizations
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], #Regularization Strengths, Lower value implies stricter Regularization
        'solver': ['liblinear', 'saga'], #Solvers
        'max_iter': [1000, 10000, 100000], #Max Number of Iterations for the solver to converge
        'class_weight': ['balanced'], #For Class imbalances
        'l1_ratio': [0.0, 0.15, 0.5, 0.85, 1.0] # L1 ratio for elasticnet
    }
    
    #Initialize the RandomSearch Cross Validator for Hyperparameter Tuning
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50, #Pick Random 50 random combinations from possible Hyperparameter combinations
        cv=stratified_kfold,
        scoring='accuracy', #User accuracy for the evaluation of hyperparameters
        n_jobs=-1, #Use all CPUs
        random_state=42
        )
    
    #Fit the Randomized CV to the Training Data
    grid_result = random_search.fit(X_train, y_train)
    
    #Print the best hyperparameters
    logger.info(f"Best result for the LR Model: {grid_result.best_score_} using {grid_result.best_params_}")
    
    #Predict on test_data
    y_pred = grid_result.best_estimator_.predict(X_test)
    
    # Map the encoded labels back to the original names
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    #Evaluate the Model using accuracy, precision, recall, f1-score
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    logger.info(f"Test Accuracy: {accuracy * 100:2f}%")
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    #Generate reports:
    logger.info("Generating Reports for the Logistic Regression Model")
    generate_report(f'LogisticRegression', classification_report(y_test_labels, y_pred_labels, zero_division=0), elapsed_time)
    
    logger.info("Reports generated and Logistic Regression Model fitted")
    
    save_model(grid_result.best_estimator_, 'lr', model_type)
    
    return grid_result.best_estimator_    

def svm(X_train, X_test, y_train, y_test, label_encoder, model_type):
    
    logger.info("Fitting the SVM Model")
    
    start_time = time.time()
    
    #Initialize Support Vector Classifier Model
    model = SVC()
    
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], #Regularization Parameter
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], #Kernel Types
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], #Kernel Coefficient
        'class_weight': ['balanced'] #For class imbalances
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='accuracy',
        cv=stratified_kfold,
        n_jobs=-1,
        random_state=42
    )
    
    grid_result = random_search.fit(X_train, y_train)
    
    logger.info(f"Best result for the SVM Model: {grid_result.best_score_} using {grid_result.best_params_}")
    
    y_pred = grid_result.best_estimator_.predict(X_test)
    
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    logger.info(f"Test Accuracy: {accuracy * 100:2f}%")
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    logger.info("Generating report for the SVM Model")
    
    generate_report(f'SVM', classification_report(y_test_labels, y_pred_labels, zero_division=0), elapsed_time)
    
    logger.info("Report generated and SVM Model fitted")

    save_model(grid_result.best_estimator_, 'svm', model_type)
    
    return grid_result.best_estimator_

def randomforest(X_train, X_test, y_train, y_test, label_encoder, model_type):
    
    logger.info("Fitting the Random Forest Model")
    
    start_time = time.time()
    
    #Initialize the Random Forest Model
    model = RandomForestClassifier()
    
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 400, 600, 800, 1000], # Number of trees in the forest
        'max_depth': [5, 10, 20, 50, 100, None], # Maximum depth of the tree
        'min_samples_split': [2, 5, 10, 20, 50], # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4, 10], # Minimum number of samples required to be at a leaf node
        'max_features': [None, 'sqrt', 'log2', 0.5], # Number of features to consider when looking for the best split
        'bootstrap': [True, False], # Whether bootstrap samples are used when building trees
        'class_weight': ['balanced'] # Weights associated with classes to handle class imbalance
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='accuracy',
        cv=stratified_kfold,
        n_jobs=-1,
        random_state=42
    )
    
    grid_result = random_search.fit(X_train, y_train)
    
    logger.info(f"Best result for the Random Forest Model: {grid_result.best_score_} using {grid_result.best_params_}")
    
    y_pred = grid_result.best_estimator_.predict(X_test)
    
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    logger.info(f"Test Accuracy: {accuracy * 100:2f}%")
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    logger.info("Generating report for the Random Forest Model")
    
    generate_report(f'RandomForest', classification_report(y_test_labels, y_pred_labels, zero_division=0), elapsed_time)
    
    logger.info("Report generated and Random Forest Model fitted")

    save_model(grid_result.best_estimator_, 'rf', model_type)
    
    return grid_result.best_estimator_

def gbm(X_train, X_test, y_train, y_test, label_encoder, model_type):
    
    logger.info("Fitting the GBM Model")
    
    start_time = time.time()
    
    #Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    
    stratified_kfold = StratifiedKFold()
    
    param_grid = {
        'n_estimators': [100, 200, 400, 600, 800, 1000, 1500], # Number of boosting stages to be run
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], # Step size shrinkage used in update to prevent overfitting
        'max_depth': [2, 3, 4, 6, 8, 10], # Maximum depth of the individual regression estimators
        'min_samples_split': [2, 5, 10, 20, 50], # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 5, 10, 20], # Minimum number of samples required to be at a leaf node
        'max_features': [None, 'sqrt', 'log2', 0.5], # Number of features to consider when looking for the best split
        'subsample': [0.5, 0.7, 0.8, 1.0] # Fraction of samples used for fitting the individual base learners
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='accuracy',
        cv=stratified_kfold,
        n_jobs=-1,
        random_state=42
    )
    
    grid_result = random_search.fit(X_train, y_train)
    
    logger.info(f"Best result for the GBC Model: {grid_result.best_score_} using {grid_result.best_params_}")

    y_pred = grid_result.best_estimator_.predict(X_test)
    
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    logger.info(f"Test Accuracy: {accuracy * 100:2f}%")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("Generating report for the GBM Model")
    
    generate_report(f'GBM', classification_report(y_test_labels, y_pred_labels, zero_division=0), elapsed_time)

    logger.info("Report generated and model fitted")
    
    save_model(grid_result.best_estimator_, 'gbm', model_type)
    
    return grid_result.best_estimator_

def model_selection(X_train, X_test, y_train, y_test, label_encoder, model_type):
    
    models = {
        'Logistic Regression': logistic_regression,
        'SVM': svm,
        'Random Forest': randomforest,
        'Gradient Boosting': gbm
    }
    
    results = {}
    
    logger.info("Starting training on all models")
    
    #Train the models
    for model_name, model_func in models.items():
        best_model = model_func(X_train, X_test, y_train, y_test, label_encoder, model_type)
        results[model_name] = best_model

    # Evaluate all models on the test set and compare their accuracies
    accuracies = {}
    
    for model_name, model in results.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[model_name] = accuracy
        logger.info(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")


    # Identify the best model
    best_model_name = max(accuracies, key=accuracies.get)
    logger.info(f"The best model is: {best_model_name} with an accuracy of {accuracies[best_model_name] * 100:.2f}%")    
    
    save_model(results[best_model_name], 'best', model_type)
    
    return results[best_model_name] 

def main(argument=None):
    
    # Check if an argument is passed directly (from training_pipeline) or via command-line
    if argument is None:
        if len(sys.argv) != 2:
            print("Please pass the correct number of Command Line Arguments")
            sys.exit(1)
        argument = sys.argv[1]
    

    
    #Load the JSON data
    data_directory = os.path.join(script_dir, 'Data', 'Processed')
    data = load_json_file(os.path.join(data_directory, 'tickets.json'))
    logger.info("Data loaded")
    #Convert it into a Pandas DataFrame object
    df = pd.DataFrame(data)
    logger.info("Data converted to a Pandas DataFrame object")
    
    #Generate Reports/Plots for the input data
    logger.info("Generating Input Data Reports")
    input_report(df, 'Category')
    input_report(df, 'Resource')
    logger.info("Input Data reports generated")


    #Prepare the Training Data
    X = df['Title'].apply(' '.join) + df['Text'].apply(' '.join) #Join the title field and text field of the ticket, words are joined back too
    y_category = df['Category']
    y_resource = df['Resource']
    
    #Encode the target labels as integers
    logger.info("Encoding the target labels")
    label_encoder_category = LabelEncoder()
    label_encoder_resource = LabelEncoder()    
    y_category_encoded = label_encoder_category.fit_transform(y_category)
    y_resource_encoded = label_encoder_resource.fit_transform(y_resource)
    logger.info("Labels encoded")
    
    #Split the data into Training/Test (80/20) test
    X_train, X_test, y_category_train, y_category_test, y_resource_train, y_resource_test = train_test_split(
    X, y_category_encoded, y_resource_encoded, test_size=0.2, random_state=42
    )
    
    #Use TF-IDF Vectorizer to extract features
    logger.info("Extracting Features using TF-IDF Vectorizer")
    vectorizer, X_train_vec, X_test_vec = tfidf_vectorizer(X_train, X_test)
    logger.info("Features extracted")
    
    # Save preprocessors
    save_preprocessors(vectorizer, label_encoder_category, label_encoder_resource)


    #Logistic Regression
    if argument == 'lr':
        logistic_regression(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        logistic_regression(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
        
    
    #SVM
    elif argument == 'svm':
        svm(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        svm(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    #RandomForest
    elif argument  == 'rf':
        randomforest(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        randomforest(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    #Gradient Boosting Machine
    elif argument == 'gbm':
        gbm(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        gbm(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    #Train using all models and select the best one
    elif argument  == 'train':
        model_selection(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        model_selection(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    else:
        print("Please pass the correct Command Line Argument")
        sys.exit(1)

if __name__ == '__main__':
    main()