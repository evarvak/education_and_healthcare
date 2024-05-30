import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feature_selection_helpers as fsh
import NSCH_helpers as nh
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, fowlkes_mallows_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def clf_metrics(clf, X, y, threshold=0.5):
    '''
    Returns metrics evaluating the performance of the classifier clf on the data X with true labels y.
    Arguments:
        clf - a fitted classifier. NOTE: must be able to predict probabilies (probability=True for most non-logistic classifiers)
        X - dataset on which clf is to be evaluated
        y - array of true labels of data contained in X
        threshold - probability threshold or sequence of thresholds, above which probabilities will be labeled type 1 (type 0 for probabilities below thresh)
    Returns:
        dataframe containing various performance metrics for each threshold in threshold
    '''
    probs = clf.predict_proba(X)
    index = ['Accuracy', 'Precision', 'Recall', 'Average_Precision_Score', 'F1 score']

    if not hasattr(threshold,'__iter__'):
        threshold = [threshold]
    column = ['Threshold = ' + str(round(thresh,3)) for thresh in threshold]
    df = pd.DataFrame(index=index,columns=column)
    for i, thresh in enumerate(threshold):
        df[column[i]] = [accuracy_score(y, (probs[:,1]>thresh)), precision_score(y, (probs[:,1]>thresh)),
                        recall_score(y, (probs[:,1]>thresh)), average_precision_score(y,probs[:,1]),
                        f1_score(y, (probs[:,1]>thresh))]
    
    return df




def optimal_threshold(precision, recall, thresholds):

    '''
    Takes in the precision, recall, and thresholds returned by sk.metrics.precision_recall_curve and determines the (precision, recall)
    pair closest to (1,1). Also returns the optimal threshold and the index of the optimal threshold in the thresholds array

    Arguments:
        precision, recall, thresholds -- output of sk.metrics.precision_recall_curves
    Returns
        precision[opt_ind] -- precision value in the pair (precision, recall) closest to the point (1,1)
        recall[opt_ind] -- recall value in the pair (precision, recall) closest to the point (1,1)
        thresholds[opt_ind] -- threshold value corresponding to the optimal (precision, recall) values above
        opt_ind -- index of the optimal threshold in the thresholds array
    '''
    opt_ind = np.argmin((precision-1)**2 + (recall-1)**2)

    return precision[opt_ind], recall[opt_ind], thresholds[opt_ind], opt_ind




def plot_precision_recall_curve(precision, recall, thresholds):

    '''
    Print the precision-recall curve from classifier predictions with the optimal threshold
    as defined in optimal_threshold() indicated

    Arguments:
        precision, recall, thresholds -- output of sk.metrics.precision_recall_curves
    Returns:
        None
    '''

    _, _, _, opt_thresh = optimal_threshold(precision, recall, thresholds)


    plt.figure()
    plt.plot(recall, precision)
    plt.plot(recall[opt_thresh], precision[opt_thresh],'ok',markersize=10)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
    
    return None


def split_impute(X, y, test_size=0.2,
                imputer = 'rf',
                has_response = False,
                test_has_response = False):
    '''
    Return the train test split with data imputed using method imputer
    Arguments:
        X - features dataframe
        y - target series
        imputer - string or None, method of imputing. Can be 'mode', 'rf', or None. 
                  If None, then no imputing will be done and all rows with missing data will be dropped. Defaults to 'rf'
    Returns:
        (X_train, X_test, y_train, y_test) or (X_train_imputed, X_test_imputed, y_train, y_test)
    '''
    assert imputer in {'rf','mode',None}, "Imputer method must be 'rf', 'mode', or None"
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)

    if imputer is not None:
        X_train_imputed = nh.impute_NSCH(X_train, imputer = imputer, state='abbr', has_response=has_response ,test_has_response = test_has_response)
        X_test_imputed = nh.impute_NSCH(X_train, imputer = imputer,test=True, test_data=X_test, state='abbr', has_response=has_response ,test_has_response = test_has_response)
        return (X_train_imputed, X_test_imputed, y_train, y_test)
    else:
        return (X_train, X_test, y_train, y_test)
