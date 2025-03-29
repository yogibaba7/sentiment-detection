import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import logging
# configure logging

logger = logging.getLogger('model_evaluation_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# read test data
def read_test(test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug(f"loading test data from {test_path}")
    try:
        test_df = pd.read_csv(test_path)
        x_test = test_df.drop(columns=['label'])
        y_test = test_df['label']
        return x_test, y_test
    except Exception as e:
        logger.error(f"Unexpected error while reading test data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# Load the model from the pickle file
def load_model(model_path: str) -> GradientBoostingClassifier:
    logger.debug(f"loading the model from {model_path}")
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        return None


# make prediction
def predict(model: GradientBoostingClassifier, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    logger.debug('Making prediction')
    try:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)
        return y_pred, y_pred_proba
    except Exception as e:
        logger.error(f"Error while making predictions: {e}")
        return np.array([]), np.array([])


# scores
def store_result(file_path: str, y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:

    try:
        logger.debug('calculating results')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_proba[:, 1])

        score_dict = {
            'accuracy_score': accuracy,
            'precision_score': precision,
            'recall_score': recall,
            'roc_score': roc
        }
        logger.debug(f"storing the result on file {file_path}")

        with open(file_path, 'w') as file:
            json.dump(score_dict, file, indent=4)

        logger.debug(f"result successfully stored on file {file_path}")

    except Exception as e:
        logger.error(f"Error while storing results: {e}")


# main
def main():
    try:
        test_path = 'data/processed/test_bow.csv'
        model_path = 'models/model.pkl'

        x_test, y_test = read_test(test_path)
        model = load_model(model_path)
        y_pred, y_pred_proba = predict(model, x_test)
        file_path = 'reports/metrics.json'
        store_result(file_path, y_test, y_pred, y_pred_proba)
        logger.debug('model evaluation successfully completed')
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


if __name__ == '__main__':
    main()