import numpy
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from amaretto_lib.classifier.Classifier import Classifier


class FNTunedClassifier(Classifier):
    """
    Basic Class for Tuned Classifiers considering a max FN threshold.
    This is useful to constraint a classifier under a specific FN threshold
    """

    def __init__(self, max_FNR: float = 0.1, clf=DecisionTreeClassifier(), tv_split: float = 0.8, normal_class=0):
        """
        Constructor of a FN Tuned Classifier (only binary classifiers)
        :param clf: algorithm to be used as Classifier
        """
        super().__init__(clf)
        self.max_FNR = max_FNR
        self.tv_split = tv_split
        self.normal_class = normal_class
        self.probability_threshold = None

    def fit(self, X, y, verbose: bool = False):
        """
        Fits the FNClassifier
        :param X: the train set
        :param y: the train labels (cannot be unsupervised)
        :return: placeholder (self)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-self.tv_split, random_state=42)
        super().fit(X_train, y_train)

        # Iterative process for FN tuning
        y_pred = super().predict(X_test)
        y_proba = super().predict_proba(X_test)
        fns = (y_test != y_pred) * (y_pred == self.normal_tag)
        fn_probas = sorted(y_proba[fns, 0])
        self.probability_threshold = 0.5
        while len(fn_probas) > 0:
            residual_FNR = len(fn_probas) / len(y_test)
            if residual_FNR < self.max_FNR:
                # Means that we are able to avoid enough FNs to comply with the max_FNR
                break
            # Otherwise, we have to modify the decision range (lower "normal" probability)
            self.probability_threshold = fn_probas.pop(0)
            while len(fn_probas) > 0 and fn_probas[0] == self.probability_threshold:
                self.probability_threshold = fn_probas.pop(0)

        if verbose:
            print("FN tuning process ended as: p(normal) > %.3f" % self.probability_threshold)

    def complies_constraint(self):
        """
        True if tuning ended successfully
        :return: a boolean
        """
        return self.probability_threshold is not None

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        dec_thr = self.probability_threshold if self.probability_threshold is not None else -1
        return self.classes_[1*(probas[:, 0] <= dec_thr)]

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return "TunedFN(" + super().classifier_name() + "- maxFNR: " + str(self.max_FNR) + ")"
