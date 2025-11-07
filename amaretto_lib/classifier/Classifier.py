import copy

import numpy
from pyod.models.base import BaseDetector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

from amaretto_lib.utils.classifier_utils import get_classifier_name
from amaretto_lib.utils.general_utils import current_ms


class Classifier(ClassifierMixin, BaseEstimator):
    """
    Basic Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, clf = DecisionTreeClassifier(), normal_tag: str = "normal"):
        """
        Constructor of a generic Classifier
        :param clf: algorithm to be used as Classifier
        """
        super().__init__()
        self.clf = copy.deepcopy(clf) if clf is not None else None
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None
        self.normal_tag = normal_tag

    def fit(self, X, y=None, verbose: bool = False):
        """
        Fits the Classifier
        :param verbose: True if debug information has to be shown
        :param X: the train set
        :param y: the train labels (can be None if unsupervised)
        :return: placeholder (self)
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        self.classes_ = list(unique_labels(y)) if y is not None else [self.normal_tag, "attack"]
        if self.is_unsupervised() and y is not None:
            # Makes sure the first tag is normal_tag
            self.classes_.remove(self.normal_tag)
            self.classes_.insert(0, self.normal_tag)
            # Checks for contamination
            att_cont = sum(1*(y != self.normal_tag)) / len(y)
            if att_cont <= 0.5:
                self.clf.contamination = att_cont
            else:
                self.clf.contamination = 0.5
                self.classes_ = self.classes_[::-1]
        self.classes_ = numpy.asarray(self.classes_)

        # Train clf
        start_ms = current_ms()
        if self.is_unsupervised():
            # Unsupervised
            self.clf.fit(X)
        else:
            # Supervised
            self.clf.fit(X, y)
        if verbose:
            print("Training completed in %d ms" % current_ms() - start_ms)

        self.feature_importances_ = self.compute_feature_importances()


    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        if isinstance(self.clf, BaseDetector):
            return self.classes_[self.clf.predict(X)]
        else:
            return self.classes_[numpy.argmax(self.predict_proba(X), axis=1)]

    def decision_function(self, X):
        """
        pyod function to override. Calls the wrapped classifier or calls decision_function function.
        :param X: test set
        :return: decision function
        """
        if self.is_unsupervised():
            return self.clf.decision_function(X)
        else:
            if X is None:
                return None
            X = check_array(X)
            probas = self.predict_proba(X)
            if probas.shape[1] >= 2:
                a = probas[:, 1] / probas[:, 0]
                return a
            else:
                return numpy.zeros(X.shape[0])

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes.
        For unsupervised, it has to be overridden since PYOD's implementation of predict_proba is wrong
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        if self.is_unsupervised():
            # Unsupervised
            pred_score = self.decision_function(X)
            probs = numpy.zeros((X.shape[0], 2))
            if isinstance(self.clf.contamination, (float, int)):
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
        else:
            # Supervised
            return self.clf.predict_proba(X)

    def predict_confidence(self, X):
        """
        Method to compute confidence in the predicted class
        :return: max probability as default
        """
        probas = self.predict_proba(X)
        return numpy.max(probas, axis=1)

    def compute_feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        elif hasattr(self.clf, 'coef_'):
            return numpy.sum(numpy.absolute(self.clf.coef_), axis=0)
        return []

    def is_unsupervised(self):
        """
        true if the classifier is unsupervised
        :return: boolean
        """
        return isinstance(self.clf, BaseDetector)

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return get_classifier_name(self.clf)

    def set_params(self, **parameters):
        """
        Compatibility with scikit-learn
        :param parameters:
        :return:
        """
        for parameter, value in parameters.items():
            setattr(self.clf, parameter, value)
        return self
