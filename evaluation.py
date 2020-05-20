"""
Models evaluation function. This is a modified version of evaluation code provided here:
https://github.com/udacity/ML_SageMaker_Studies/blob/master/Payment_Fraud_Detection/Fraud_Detection_Solution.ipynb
"""

import numpy as np
import pandas as pd


def evaluate(offers, verbose=True, csv=False, jsonlines=False, beta=1):
    """
    Evaluate a model on a test set given the prediction endpoint or output files.
    Return binary classification metrics (recall, precision, accuracy, F-score).

    :param offers: list. List of dictionaries, each of which contains a prediction endpoint (or location of
                   output files), test features and labels
    :param verbose: bool. If True, prints a table of all performance metrics
    :param csv: bool. If model's predictions are saved as csv-files
    :param jsonlines: bool. If model's predictions are saved as jsonlines-files
    :param beta: float. Beta value for F-score metric (F1 when beta is 1)
    :return: dict. Model's performance metrics.
    """
    tp = fp = tn = fn = 0
    for offer in offers:
        if csv:
            test_preds = pd.read_csv(offer['location'], header=None).squeeze().values.round()
        elif jsonlines:
            test_preds = pd.read_json(offer['location'], orient='records', lines=True)
            test_preds = test_preds['predicted_label'].squeeze().values
        else:   # prediction endpoint
            # split the test data set into batches and evaluate using prediction endpoint
            prediction_batches = [offer['predictor'].predict(batch) for batch in np.array_split(offer['X'], 100)]

            # LinearLearner produces a `predicted_label` for each data point in a batch
            test_preds = np.concatenate([np.array([x.label['predicted_label'].float32_tensor.values[0] for x in batch])
                                         for batch in prediction_batches])

        # calculate true positives, false positives, true negatives, false negatives
        tp += np.logical_and(offer['y'], test_preds).sum()
        fp += np.logical_and(1 - offer['y'], test_preds).sum()
        tn += np.logical_and(1 - offer['y'], 1 - test_preds).sum()
        fn += np.logical_and(offer['y'], 1 - test_preds).sum()

        if verbose:
            print(offer['name'] + ' offers')
            print(pd.crosstab(offer['y'], test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
            print('\n')

    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f_score = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)

    # print a table of metrics
    if verbose:
        print('Total:')
        print(f"{'Recall:':<11} {recall:.3f}")
        print(f"{'Precision:':<11} {precision:.3f}")
        print(f"{'Accuracy:':<11} {accuracy:.3f}")
        print(f"{'F-score:':<11} {f_score:.3f}")

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy, 'F-score': f_score}
