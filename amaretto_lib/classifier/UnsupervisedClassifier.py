import numpy
from pyod.models.base import BaseDetector
from sklearn.utils.validation import check_is_fitted, check_array

from amaretto_lib.classifier.Classifier import Classifier


class UnsupervisedClassifier(Classifier, BaseDetector):
    """
    Wrapper for unsupervised classifiers belonging to the library PYOD
    """

    def __init__(self, clf):
        """
        Constructor of a generic UnsupervisedClassifier. Assumes that clf is an algorithm from pyod
        :param clf: pyod algorithm to be used as Classifier
        """
        super().__init__(clf)
        self.contamination = clf.contamination

    def fit(self, X, y=None):
        # Store the classes seen during fit + other data
        self.classes_ = [0, 1]
        self.X_ = X
        self.y_ = None

        # Train clf
        self.clf.fit(X)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def decision_function(self, X):
        """
        pyod function to override. Calls the wrapped classifier.
        :param X: test set
        :return: decision function
        """
        return self.clf.decision_function(X)

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes.
        It has to e overridden since PYOD's implementation of predict_proba is wrong
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        pred_score = self.decision_function(X)
        probs = numpy.zeros((X.shape[0], 2))
        if isinstance(self.contamination, (float, int)):
            pred_thr = pred_score - self.clf.threshold_
        min_pt = min(pred_thr)
        max_pt = max(pred_thr)
        anomaly = pred_thr > 0
        cont = numpy.asarray([pred_thr[i] / max_pt if anomaly[i] else (pred_thr[i] / min_pt if min_pt != 0 else 0.2)
                              for i in range(0, len(pred_thr))])
        probs[:, 0] = 0.5 + cont / 2
        probs[:, 1] = 1 - probs[:, 0]
        probs[anomaly, 0], probs[anomaly, 1] = probs[anomaly, 1], probs[anomaly, 0]
        return probs

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return self.clf.__class__.__name__