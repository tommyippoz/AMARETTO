import collections
import copy

import numpy
from numpy import ndarray
from pyod.models.base import BaseDetector
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from amaretto_lib.classifier.Classifier import Classifier


class IntrusionDetector(Classifier):

    def __init__(self,
                 signature_clf=ExtraTreeClassifier(),
                 binary_clf=RandomForestClassifier(),
                 multi_clf=RandomForestClassifier(),
                 zeroday_clf=IForest(contamination=0.1),
                 normal_tag: str = "normal", zeroday_tag: str = "zeroday"):
        super().__init__()
        self.normal_tag = normal_tag
        self.zeroday_tag = zeroday_tag
        self.signature_clf = signature_clf
        self.binary_clf = binary_clf
        self.multi_clf = multi_clf
        self.zeroday_clf = zeroday_clf
        self.classes_ = None
        self.attacks_ = None

    def fit(self, X, y):
        """
        Fits the Intrusion Detector
        :param X: the train set
        :param y: the train labels
        :return: a placeholder (self)
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        y_bin = numpy.where(y == self.normal_tag, self.normal_tag, "attack")

        # Store the classes seen during fit + attacks
        self.classes_ = unique_labels(y)
        self.attacks_ = [x for x in self.classes_ if x != self.normal_tag]

        # Train signature clfs (1), gets a dict
        self.signature_clfs = self.fit_signature_clfs(X, y)

        # Train binary clf (2)
        if not is_classifier(self.binary_clf):
            self.binary_clf = RandomForestClassifier()
        self.binary_clf.fit(X, y_bin)

        # Train unsupervised binary clf (4)
        contamination = collections.Counter(y_bin)["attack"] / len(y_bin)
        if not isinstance(self.zeroday_clf, BaseDetector):
            self.zeroday_clf = COPOD()
        self.zeroday_clf.contamination = contamination if contamination <= 0.5 else 0.5
        self.zeroday_clf.fit(X)

        # Train multi-class classifier on data points that were labeled as attacks by the bin_clf
        att_mask = (1*(y != self.normal_tag) + 1*(y != self.binary_clf.predict(X))) > 0
        if not is_classifier(self.multi_clf):
            self.multi_clf = RandomForestClassifier()
        self.multi_clf.fit(X[att_mask, :], y[att_mask])

        # Return the classifier
        return self

    def fit_signature_clfs(self, X: ndarray, y: ndarray) -> dict:
        """
        This function returns a dictionary containing entries for each signature-based strategy
        :param X: the train set
        :param y: the train labels
        :return: a dictionary
        """
        s_clfs = {}
        for attack_tag in self.attacks_:
            att_mask = (1*(y == self.normal_tag) + 1*(y == attack_tag)) > 0
            if is_classifier(self.signature_clf):
                clf = copy.deepcopy(self.signature_clf)
            elif isinstance(self.signature_clf, dict) and attack_tag in self.signature_clf:
                clf = copy.deepcopy(self.signature_clf[attack_tag])
            else:
                clf = ExtraTreeClassifier()
            clf.fit(X[att_mask, :], y[att_mask])
            s_clfs[attack_tag] = clf
        return s_clfs

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        predictions = {}
        for attack_tag in self.signature_clfs:
            predictions[attack_tag] = self.signature_clfs[attack_tag].predict(X)
        predictions["binary"] = self.binary_clf.predict(X)
        predictions["multi"] = self.multi_clf.predict(X)
        predictions["zeroday"] = self.zeroday_clf.predict(X)
        result = [None for _ in range(0, X.shape[0])]
        for i in range(0, len(result)):
            # First we ask signature-based methods
            for attack_tag in self.signature_clfs:
                if predictions[attack_tag][i] != self.normal_tag:
                    result[i] = attack_tag
                    break
            if result[i] is None:
                # This means that signature-based methods did not find anything
                if predictions["binary"][i] == self.normal_tag:
                    # This means that the binary classifies did not find attacks
                    # Thus, responsability is up to the zero-day detector
                    result[i] = self.normal_tag if predictions["zeroday"][i] == 0 else self.zeroday_tag
                else:
                    # We run multi-class for diagnosis
                    # Also, it potentially spots FPs of the binary and corrects them
                    result[i] = predictions["multi"][i]

        return numpy.asarray(result)

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """

        return None
