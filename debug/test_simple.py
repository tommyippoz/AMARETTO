# Support libs
import os
import random
import time

import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms
# Used to save a classifier and measure its size in KB
from pyod.models.iforest import IForest
from sklearn.base import is_classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Scikit-Learn algorithms
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting

from amaretto_lib.classifier.Classifier import Classifier

# ------- GLOBAL VARS -----------
from amaretto_lib.classifier.FNTunedClassifier import FNTunedClassifier
from amaretto_lib.classifier.FPTunedClassifier import FPTunedClassifier
from amaretto_lib.classifier.IntrusionDetector import IntrusionDetector
from amaretto_lib.utils.classifier_utils import get_classifier_name

CSV_FOLDER = "input_folder"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "test_out.csv"
# Percantage of test data wrt train data
TT_SPLIT = 0.6
# True if debug information needs to be shown
VERBOSE = True
# mostly for debug
MAKE_BINARY = True


# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


# --------- SUPPORT FUNCTIONS ---------------


def current_milli_time():
    """
    gets the current time in ms
    :return: a long int
    """
    return round(time.time() * 1000)


def get_learners(cont_perc):
    """
    Function to get a learner to use, given its string tag
    :param cont_perc: percentage of anomalies in the training set, required for unsupervised classifiers from PYOD
    :return: the list of classifiers to be trained
    """
    return [
        #DecisionTreeClassifier(),
        #IntrusionDetector(),
        #RandomForestClassifier(),
        #FNTunedClassifier(max_FNR=0.0003, clf=RandomForestClassifier()),
        Classifier(clf=IForest(contamination=0.3)),
        FNTunedClassifier(max_FNR=0.1, clf=IForest(contamination=0.3)),
        FNTunedClassifier(max_FNR=0.01, clf=IForest(contamination=0.3)),
        FNTunedClassifier(max_FNR=0.001, clf=IForest(contamination=0.3)),
        FNTunedClassifier(max_FNR=0.0001, clf=IForest(contamination=0.3)),
        FPTunedClassifier(max_FPR=0.1, clf=IForest(contamination=0.3)),
        FPTunedClassifier(max_FPR=0.01, clf=IForest(contamination=0.3)),
        #ConfidenceBagging(clf=ExtraTreeClassifier(), n_base=10),
        #ConfidenceBoosting(clf=RandomForestClassifier(n_estimators=5), n_base=10),

    ]

# ----------------------- MAIN ROUTINE ---------------------


if __name__ == '__main__':

    existing_exps = None
    # if os.path.exists(SCORES_FILE):
    #     existing_exps = pandas.read_csv(SCORES_FILE)
    #     existing_exps = existing_exps.loc[:, ['dataset_tag', 'clf']]
    # else:
    with open(SCORES_FILE, 'w') as f:
        f.write("dataset_tag,clf,binary,tt_split,acc,misc,mcc,bacc,time,model_size\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):

            # if file is a CSV, it is assumed to be a dataset to be processed
            df = pandas.read_csv(full_name, sep=",")
            df = df.sample(frac=1.0)
            if len(df.index) > 100000:
                df = df.iloc[:100000, :]
            if VERBOSE:
                print("\n------------ DATASET INFO -----------------")
                print("Data Points in Dataset '%s': %d" % (dataset_file, len(df.index)))
                print("Features in Dataset: " + str(len(df.columns)))

            # Filling NaN and Handling (Removing) constant features
            df = df.fillna(0)
            df = df.loc[:, df.nunique() > 1]
            if VERBOSE:
                print("Features in Dataframe after removing constant ones: " + str(len(df.columns)))

            features_no_cat = df.select_dtypes(exclude=['object']).columns
            if VERBOSE:
                print("Features in Dataframe after non-numeric ones (including label): " + str(len(features_no_cat)))

            # Binarize if needed (for anomaly detection you need a 2-class problem, requires one of the classes to have NORMAL_TAG)
            normal_perc = None
            y = df[LABEL_NAME].to_numpy()
            if MAKE_BINARY:
                y = numpy.where(y == NORMAL_TAG, NORMAL_TAG, "attack")
            if VERBOSE:
                print("Dataset contains %d Classes" % len(numpy.unique(y)))


            # Set up train test split excluding categorical values that some algorithms cannot handle
            # 1-Hot-Encoding or other approaches may be used instead of removing
            x_no_cat = df.select_dtypes(exclude=['object']).to_numpy()
            x_train, x_test, y_train, y_test = ms.train_test_split(x_no_cat, y, test_size=TT_SPLIT, shuffle=True)

            if VERBOSE:
                print('-------------------- CLASSIFIERS -----------------------')

            # Loop for training and testing each learner specified by LEARNER_TAGS
            contamination = 1 - normal_perc if normal_perc is not None else None
            learners = get_learners(contamination)
            i = 1
            for classifier in learners:

                # Getting classifier Name
                clf_name = get_classifier_name(classifier)
                if existing_exps is not None and (((existing_exps['dataset_tag'] == full_name) &
                                                   (existing_exps['clf'] == clf_name)).any()):
                    print('%d/%d Skipping classifier %s, already in the results' % (i, len(learners), clf_name))

                elif is_classifier(classifier):
                    # Training the algorithm to get a mode
                    start_time = current_milli_time()
                    classifier.fit(x_train, y_train)
                    train_time = current_milli_time() - start_time

                    # Quantifying size of the model
                    # dump(classifier, "clf_d.bin", compress=9)
                    size = 0  # os.stat("clf_d.bin").st_size
                    # os.remove("clf_d.bin")

                    # Scoring
                    y_pred = classifier.predict(x_test)
                    #y_proba = classifier.predict_proba(x_test)
                    #y_conf = numpy.max(y_proba, axis=1)

                    # Computing Metrics
                    acc = metrics.accuracy_score(y_test, y_pred)
                    misc = int((1 - acc) * len(y_test))
                    mcc = abs(metrics.matthews_corrcoef(y_test, y_pred))
                    bacc = metrics.balanced_accuracy_score(y_test, y_pred)

                    # Prints just accuracy for multi-class classification problems, no confusion matrix
                    print('%d/%d Accuracy: %.3f, MCC: %.3f, train time: %d \t-> %s' %
                          (i, len(learners), acc, mcc, train_time, clf_name))

                    # Updates CSV file form metrics of experiment
                    with open(SCORES_FILE, "a") as myfile:
                        # Prints result of experiment in CSV file
                        myfile.write(full_name + "," + clf_name + "," + str("NOPE") + "," +
                                         str(TT_SPLIT) + ',' + str(acc) + "," + str(misc) + "," + str(mcc) + "," +
                                         str(bacc) + "," + str(train_time) + "," + str(size) + "\n")

                classifier = None
                i += 1
