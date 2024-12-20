from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier
from cuml import TfidfVectorizer
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
import warnings
import cupy as cp
from cuml.dask.common import to_dask_cudf
import dask_cudf
import cudf

script_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize logger and configure the directories/logger 
current_date = datetime.now().strftime('%Y-%m-%d')
date_dir = os.path.join(script_dir, '..', 'Monitoring', 'logs', current_date)
os.makedirs(date_dir, exist_ok=True)

log_file_path = os.path.join(date_dir, 'gputraining_script.log')
logger = logging.getLogger('gputraining_logger')
logger.setLevel(logging.INFO)

fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)

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
    
    plt.figure(figsize=(16, 20))
    ax = count.plot(kind='barh')
    plt.xlabel('Count')
    plt.ylabel(f'{type}')
    plt.title(f'{type} Counts')

    for index, value in enumerate(count):
        plt.text(value, index, str(value))

    plt.savefig(os.path.join(date_dir, f'{type}_counts_plot.png'), dpi=300, bbox_inches='tight')

def generate_report(model, classification_report, elapsed_time):
    with open(os.path.join(date_dir, f'{str(model)}_report.txt'), 'a') as file:
        file.write("#" * 200 + "\n")
        file.write("Report generated at: " + datetime.now().strftime('%H-%M-%S') + "\n")
        file.write(classification_report + '\n')
        file.write("Elapsed Time: " +  str(elapsed_time) + ' seconds' + "\n")

def save_model(model, model_name, model_type):
    models_dir = os.path.join(script_dir, 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    if model_name == 'best':
        filename = f'model_{model_type}.joblib'
    else:
        filename = f'{model_name}_{model_type}.joblib'
    
    model_path = os.path.join(models_dir, filename)
    dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def save_preprocessors(vectorizer, label_encoder_category, label_encoder_resource):
    models_dir = os.path.join(script_dir, 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    dump(vectorizer, vectorizer_path)
    logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")
    
    category_encoder_path = os.path.join(models_dir, 'label_encoder_category.joblib')
    resource_encoder_path = os.path.join(models_dir, 'label_encoder_resource.joblib')
    dump(label_encoder_category, category_encoder_path)
    dump(label_encoder_resource, resource_encoder_path)
    logger.info("Label encoders saved")

def tfidf_vectorizer(train_set, test_set):
    custom_stopwords = ['misc', 'org', 'loc', 'der', 'zu', 'und', 'sein', 'in', 'auf', 'per','an', 'für', 'ich', 'ipk','grüß','von','mit', 'können', 'nach', 'werden']

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        max_df=0.8,
        min_df=0.01,
        smooth_idf=True,
        sublinear_tf=True,
        norm='l2',
        stop_words=custom_stopwords
    )
    
    # Convert to cuDF Series for GPU processing
    train_series = cudf.Series(train_set)
    test_series = cudf.Series(test_set)
    
    X_train_vec = vectorizer.fit_transform(train_series)
    X_test_vec = vectorizer.transform(test_series)

    return vectorizer, X_train_vec, X_test_vec

def logistic_regression(X_train, X_test, y_train, y_test, label_encoder, model_type):
    logger.info("Fitting the Logistic Regression Model")
    
    start_time = time.time()
    
    model = LogisticRegression(handle=cp.cuda.current_stream().ptr)
    
    # Convert to cuDF for GPU processing
    y_train_gpu = cudf.Series(y_train)
    
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 1000, 10000]
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=2,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )
    
    grid_result = random_search.fit(X_train, y_train_gpu)
    
    logger.info(f"Best result for the LR Model: {grid_result.best_score_} using {grid_result.best_params_}")
    
    y_pred = grid_result.best_estimator_.predict(X_test)
    
    # Convert predictions back to CPU for evaluation
    y_pred_cpu = y_pred.values.get() if hasattr(y_pred, 'values') else y_pred
    y_pred_labels = label_encoder.inverse_transform(y_pred_cpu)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    logger.info(f"Test Accuracy: {accuracy * 100:2f}%")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    generate_report(f'LogisticRegression', classification_report(y_test_labels, y_pred_labels), elapsed_time)
    
    save_model(grid_result.best_estimator_, 'lr', model_type)
    
    return grid_result.best_estimator_

def svm(X_train, X_test, y_train, y_test, label_encoder, model_type):
    logger.info("Fitting the SVM Model")
    
    start_time = time.time()
    
    model = SVC(handle=cp.cuda.current_stream().ptr)
    
    # Convert to cuDF for GPU processing
    y_train_gpu = cudf.Series(y_train)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=5,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )
    
    grid_result = random_search.fit(X_train, y_train_gpu)
    
    y_pred = grid_result.best_estimator_.predict(X_test)
    
    # Convert predictions back to CPU for evaluation
    y_pred_cpu = y_pred.values.get() if hasattr(y_pred, 'values') else y_pred
    y_pred_labels = label_encoder.inverse_transform(y_pred_cpu)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    logger.info(f"Test Accuracy: {accuracy * 100:2f}%")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    generate_report(f'SVM', classification_report(y_test_labels, y_pred_labels), elapsed_time)
    
    save_model(grid_result.best_estimator_, 'svm', model_type)
    
    return grid_result.best_estimator_

def randomforest(X_train, X_test, y_train, y_test, label_encoder, model_type):
    logger.info("Fitting the Random Forest Model")
    
    start_time = time.time()
    
    model = RandomForestClassifier(handle=cp.cuda.current_stream().ptr)
    
    # Convert to cuDF for GPU processing
    y_train_gpu = cudf.Series(y_train)
    
    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'max_depth': [int(x) for x in np.linspace(start=10, stop=100, num=10)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=5,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )
    
    grid_result = random_search.fit(X_train, y_train_gpu)
    
    y_pred = grid_result.best_estimator_.predict(X_test)
    
    # Convert predictions back to CPU for evaluation
    y_pred_cpu = y_pred.values.get() if hasattr(y_pred, 'values') else y_pred
    y_pred_labels = label_encoder.inverse_transform(y_pred_cpu)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    logger.info(f"Test Accuracy: {accuracy * 100:2f}%")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    generate_report(f'RandomForest', classification_report(y_test_labels, y_pred_labels), elapsed_time)
    
    save_model(grid_result.best_estimator_, 'rf', model_type)
    
    return grid_result.best_estimator_

def model_selection(X_train, X_test, y_train, y_test, label_encoder, model_type):
    models = {
        'Logistic Regression': logistic_regression,
        'SVM': svm,
        'Random Forest': randomforest
    }
    
    results = {}
    
    logger.info("Starting training on all models")
    
    for model_name, model_func in models.items():
        best_model = model_func(X_train, X_test, y_train, y_test, label_encoder, model_type)
        results[model_name] = best_model

    accuracies = {}
    
    for model_name, model in results.items():
        y_pred = model.predict(X_test)
        # Convert predictions back to CPU for evaluation
        y_pred_cpu = y_pred.values.get() if hasattr(y_pred, 'values') else y_pred
        accuracy = accuracy_score(y_test, y_pred_cpu)
        accuracies[model_name] = accuracy
        logger.info(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")

    best_model_name = max(accuracies, key=accuracies.get)
    logger.info(f"The best model is: {best_model_name} with an accuracy of {accuracies[best_model_name] * 100:.2f}%")    
    
    save_model(results[best_model_name], 'best', model_type)
    
    return results[best_model_name]

def main(argument=None):
    if argument is None:
        if len(sys.argv) != 2:
            print("Please pass the correct number of Command Line Arguments")
            sys.exit(1)
        argument = sys.argv[1]
    
    data_directory = os.path.join(script_dir, 'Data', 'Processed')
    data = load_json_file(os.path.join(data_directory, 'tickets.json'))
    logger.info("Data loaded")
    
    df = pd.DataFrame(data)
    logger.info("Data converted to a Pandas DataFrame object")
    
    logger.info("Generating Input Data Reports")
    input_report(df, 'Category')
    input_report(df, 'Resource')
    logger.info("Input Data reports generated")

    X = df['Title'].apply(' '.join) + df['Text'].apply(' '.join)
    y_category = df['Category']
    y_resource = df['Resource']
    
    logger.info("Encoding the target labels")
    label_encoder_category = LabelEncoder()
    label_encoder_resource = LabelEncoder()    
    y_category_encoded = label_encoder_category.fit_transform(y_category)
    y_resource_encoded = label_encoder_resource.fit_transform(y_resource)
    logger.info("Labels encoded")
    
    X_train, X_test, y_category_train, y_category_test, y_resource_train, y_resource_test = train_test_split(
        X, y_category_encoded, y_resource_encoded, test_size=0.2, random_state=42
    )
    
    logger.info("Extracting Features using TF-IDF Vectorizer")
    vectorizer, X_train_vec, X_test_vec = tfidf_vectorizer(X_train, X_test)
    logger.info("Features extracted")
    
    # Save preprocessors
    save_preprocessors(vectorizer, label_encoder_category, label_encoder_resource)

    if argument == 'lr':
        logistic_regression(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        logistic_regression(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    elif argument == 'svm':
        svm(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        svm(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    elif argument == 'rf':
        randomforest(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        randomforest(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    elif argument == 'train':
        model_selection(X_train_vec, X_test_vec, y_category_train, y_category_test, label_encoder_category, 'category')
        model_selection(X_train_vec, X_test_vec, y_resource_train, y_resource_test, label_encoder_resource, 'resource')
    
    else:
        print("Please pass the correct Command Line Argument")
        sys.exit(1)

if __name__ == '__main__':
    # Initialize CUDA device
    cp.cuda.runtime.setDevice(0)
    logger.info(f"Using GPU: {cp.cuda.runtime.getDevice()}")
    
    
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        logger.info("GPU memory cleaned up")
        logger.removeHandler(fh)
