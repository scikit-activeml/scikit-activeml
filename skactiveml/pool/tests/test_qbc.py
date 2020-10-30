import numpy as np
import unittest
import warnings

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.ensemble import BaggingClassifier
from skactiveml.classifier import PWC
from skactiveml.utils import rand_argmax

from skactiveml.pool import QBC

class TestQBC(unittest.TestCase):

    def setUp(self):
        self.random_state = 42
        self.X_cand = np.array([[8,1,6,8],[9,1,6,5],[5,1,6,5]])
        self.X = np.array([[1,2,5,9],[5,8,4,6],[8,4,5,9],[5,4,8,5]])
        self.y = np.array([0,0,1,1])
        self.classes = np.array([0,1])
        pass


    def test_query(self):
        self.assertRaises(ValueError, QBC, clf=GaussianProcessClassifier(), method ='this_method_does_not_exist')
        self.assertRaises(TypeError, QBC, clf=None, method='vote_entropy')
        self.assertRaises(TypeError, QBC, clf=None, method='KL_divergence')
        
        clf = PWC(random_state=self.random_state, classes=self.classes)
        ensemble = BaggingClassifier(base_estimator=clf,random_state=self.random_state)
        # KL_divergence
        qbc = QBC(clf=clf, method='KL_divergence', random_state=self.random_state)
        best_indices, utilities = qbc.query(self.X_cand, self.X, self.y, return_utilities=True)

        ensemble.fit(self.X, self.y)
        val_utilities = np.array([average_KL_divergence(ensemble, self.X_cand)])
        val_best_indices = rand_argmax(val_utilities, axis=1, random_state=self.random_state)

        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        np.testing.assert_array_equal(best_indices, val_best_indices)
        np.testing.assert_array_equal(utilities, val_utilities)

        # vote_entropy
        qbc = QBC(clf=clf, method='vote_entropy', random_state=self.random_state)
        best_indices, utilities = qbc.query(self.X_cand, self.X, self.y, return_utilities=True, random_state=self.random_state)

        ensemble.fit(self.X, self.y)
        val_utilities = np.array([vote_entropy(ensemble, self.X_cand, self.classes)])
        val_best_indices = rand_argmax(val_utilities, axis=1, random_state=self.random_state)

        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        np.testing.assert_array_equal(best_indices, val_best_indices)
        np.testing.assert_array_equal(utilities, val_utilities)


def average_KL_divergence(ensemble, X_cand):
    estimators = ensemble.estimators_
    com_probas = np.zeros((len(estimators),len(X_cand),ensemble.n_classes_))
    for i, e in enumerate(estimators):
        com_probas[i,:,:] = e.predict_proba(X_cand)

    consensus = np.sum(com_probas, axis=0)/len(estimators)
    scores = np.zeros((len(X_cand)))
    for i, x in enumerate(X_cand):
        for c in range(len(estimators)):
            for y in range(ensemble.n_classes_):
                with np.errstate(divide='ignore', invalid='ignore'):
                    if com_probas[c,i,y] != 0.0:
                        scores[i] += com_probas[c,i,y]*np.log(com_probas[c,i,y]/consensus[i,y])
    scores = scores/ensemble.n_classes_
    return scores


def vote_entropy(ensemble, X_cand, classes):
    estimators = ensemble.estimators_
    votes = np.zeros((len(X_cand), len(estimators)))
    for i, model in enumerate(estimators):
        votes[:, i] = model.predict(X_cand)

    vote_count = np.zeros((len(X_cand), len(classes)))
    for i in range(len(X_cand)):
        for c in range(len(classes)):
            for m in range(len(estimators)):
                vote_count[i,c] += (votes[i,m] == c)
        
    vote_entropy = np.zeros(len(X_cand))
    for i in range(len(X_cand)):
        for c in range(len(classes)):
            if vote_count[i,c]!=0:
                a = vote_count[i,c]/len(estimators)
                vote_entropy[i] += a*np.log(a)
    vote_entropy *= -1/np.log(len(estimators))
        
    return vote_entropy


if __name__ == '__main__':
    unittest.main()
