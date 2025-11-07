import numpy
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from amaretto_lib.classifier.Classifier import Classifier


class FPTunedClassifier(Classifier):
    """
    Basic Class for Tuned Classifiers considering a max FN threshold.
    This is useful to constraint a classifier under a specific FN threshold
    """

    def __init__(self, max_FPR: float = 0.1, clf=DecisionTreeClassifier(), tv_split: float = 0.8, normal_class=0):
        """
        Constructor of a FP Tuned Classifier (only binary classifiers)
        :param clf: algorithm to be used as Classifier
        """
        super().__init__(clf)
        self.max_FPR = max_FPR
        self.tv_split = tv_split
        self.normal_class = normal_class
        self.probability_threshold = None
        self.tuning_successful = False

    def fit(self, X, y, verbose: bool = False):
        """
        Fits the FNClassifier
        :param X: the train set
        :param y: the train labels (cannot be unsupervised)
        :return: placeholder (self)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-self.tv_split, random_state=42)
        super().fit(X_train, y_train)

        if len(self.classes_) != 2:
            print("--------------- NOT A BINARY CLASSIFIER, cannot perform FP Tuning")

        # Iterative process for FN tuning
        y_pred = super().predict(X_test)
        y_proba = super().predict_proba(X_test)
        fps = (y_test != y_pred) * (y_pred != self.normal_tag)
        fp_probas = sorted(y_proba[fps, 1])
        self.probability_threshold = 0.5
        if len(fp_probas) / len(y_test) < self.max_FPR:
            self.tuning_successful = True
        while len(fp_probas) > 0:
            residual_FPR = len(fp_probas) / len(y_test)
            if residual_FPR < self.max_FPR:
                # Means that we are able to avoid enough FPs to comply with the max_FPR
                self.tuning_successful = True
                break
            # Otherwise, we have to modify the decision range (lower "normal" probability)
            self.probability_threshold = fp_probas.pop(0)
            while len(fp_probas) > 0 and fp_probas[0] == self.probability_threshold:
                self.probability_threshold = fp_probas.pop(0)

        if verbose:
            print("FP tuning process ended as: p(attack) > %.3f" % self.probability_threshold)

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        dec_thr = self.probability_threshold if self.probability_threshold is not None else -1
        return self.classes_[1*(probas[:, 1] <= dec_thr)]

    def complies_constraint(self):
        """
        True if tuning ended successfully
        :return: a boolean
        """
        return self.tuning_successful

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return "TunedFP(" + super().classifier_name() + "- maxFPR: " + str(self.max_FPR) + ")"
