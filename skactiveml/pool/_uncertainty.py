from sklearn.utils import check_array
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression#, _logistic_loss
from ..classifier import PWC
from ..base import SingleAnnotPoolBasedQueryStrategy, ClassFrequencyEstimator
from ..utils import rand_argmax, is_labeled, MISSING_LABEL
from scipy.optimize import minimize_scalar, minimize, LinearConstraint
from scipy.interpolate import griddata
import numpy as np
import warnings


class UncertaintySampling(SingleAnnotPoolBasedQueryStrategy):
    """
    Uncertainty Sampling query stratagy.

    Parameters
    ----------
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    classes : array-like, shape=(n_classes), (default=None)
        Holds the label for each class.
    method : string (default='margin_sampling')
        The method to calculate the uncertainty, entropy, least_confident, margin_sampling, expected_average_precision and epistemic are possible.
        Epistemic only works with Parzen Window Classifier or Logistic Regression.
    precompute : boolean (default=False)
        Whether the epistemic uncertainty should be precomputed.
    missing_label : scalar | str | None | np.nan, (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label.
        Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        The random state to use.

    Attributes
    ----------
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    method : string
        The method to calculate the uncertainty. Only entropy, least_confident, margin_sampling, expected_average_precisionare and epistemic.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    precompute : boolean (default=False)
        Whether the epistemic uncertainty should be precomputed.
    missing_label : scalar | str | None | np.nan, (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label.
        Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        Random state to use.

    Methods
    -------
    query(X_cand, X, y, return_utilities=False, **kwargs)
        Queries the next instance to be labeled.

    Refereces
    ---------
    [1] Settles, Burr. Active learning literature survey.
        University of Wisconsin-Madison Department of Computer Sciences, 2009.
        http://www.burrsettles.com/pub/settles.activelearning.pdf

    [2] Wang, Hanmo, et al. "Uncertainty sampling for action recognition
        via maximizing expected average precision."
        IJCAI International Joint Conference on Artificial Intelligence. 2018.

    [3] Nguyen, Vu-Linh, Sébastien Destercke, and Eyke Hüllermeier.
        "Epistemic uncertainty sampling." International Conference on
        Discovery Science. Springer, Cham, 2019.
    """
    def __init__(self, clf, classes=None, method='margin_sampling', precompute=False, missing_label=MISSING_LABEL, random_state=None):
        super().__init__(random_state=random_state)




        self.missing_label = missing_label
        self.method = method
        self.classes = classes
        self.clf = clone(clf)
        self.precompute = precompute
        self.precomp = None



    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand : np.ndarray
            The unlabeled pool from which to choose.
        X : np.ndarray
            The labeled pool used to fit the classifier.
        y : np.array
            The labels of the labeled pool X.
        return_utilities : bool (default=False
            If True, the utilities are returned.

        Returns
        -------
        np.ndarray (shape=1)
            The index of the queried instance.
        np.ndarray  (shape=(1xlen(X_cnad))
            The utilities of all instances of X_cand(if return_utilities=True).
        """

        # validation:
        if (self.method != 'entropy' and self.method != 'least_confident' and
            self.method != 'margin_sampling' and
            self.method != 'expected_average_precision'and
            self.method != 'epistemic'):
            warnings.warn("The method '" + self.method + "' does not exist,"
                          ",'margin_sampling' will be used.")
            self.method = 'margin_sampling'

        if self.method == 'expected_average_precision' and self.classes is None:
            raise ValueError('\'classes\' has to be specified')

        # for pwc:
        if self.method == 'epistemic':
            if isinstance(clf, ClassFrequencyEstimator):
                self.method = 'epistemic_pwc'
            elif isinstance(clf, LogisticRegression):
                self.method = 'epistemic_logreg'
            else:
                raise TypeError("'clf' must be a subclass of ClassFrequencyEstimator or LogisticRegression")

        if self.precompute and self.precomp is None:
            self.precomp = np.full((2,2), np.nan)
        X, y, X_cand = check_X_y(X, y, X_cand, force_all_finite=False)


        # fit the classifier and get the probabilities
        mask_labeled = is_labeled(y, self.missing_label)
        self.clf.fit(X[mask_labeled], y[mask_labeled])
        probas = self.clf.predict_proba(X_cand)

        # caculate the utilities
        with np.errstate(divide='ignore'):
            if self.method == 'least_confident':
                utilities = -np.max(probas, axis=1)
            elif self.method == 'margin_sampling':
                sort_probas = np.sort(probas, axis=1)
                utilities = sort_probas[:,-2] - sort_probas[:,-1]
            elif self.method == 'entropy':
                utilities = -np.sum(probas * np.log(probas), axis=1)
            elif self.method == 'expected_average_precision':
                utilities = expected_average_precision(X_cand, self.classes, probas)
            elif self.method == 'epistemic_pwc':
                utilities, self.precomp = epistemic_uncertainty_pwc(self.clf, X_cand, self.precomp)
            elif self.method == 'epistemic_logreg':
                utilities = epistemic_uncertainty_logreg(X[mask_labeled], y[mask_labeled], clf, probas)

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand))
        best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


# expected average precision:
def expected_average_precision(X_cand, classes, probas):
    """
    Calculate the expected average precision.

    Parameters
    ----------
    X_cand : np.ndarray
        The unlabeled pool for which to calculated the expected average
        precision.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    proba : np.ndarray, shape=(n_X_cand, n_classes)
        The probabiliti estimation for each classes and all instance in X_cand.

    Returns
    -------
    score : np.ndarray, shape=(n_X_cand)
        The expected average precision score of all instances in X_cand.
    """
    score = np.zeros(len(X_cand))
    for i in range(len(classes)):
        for j, x in enumerate(X_cand):
            # The i-th column of p without p[j,i]
            p = probas[:,i]
            p = np.delete(p,[j])
            # Sort p in descending order
            p = np.flipud(np.sort(p, axis=0))
            
            # calculate g_arr
            g_arr = np.zeros((len(p),len(p)))
            for n in range(len(p)):
                for h in range(n+1):
                    g_arr[n,h] = _g(n, h, p, g_arr)
            
            # calculate f_arr
            f_arr = np.zeros((len(p)+1,len(p)+1))
            for a in range(len(p)+1):
                for b in range(a+1):
                    f_arr[a,b] = _f(a, b, p, f_arr, g_arr)
            
            # calculate score
            for t in range(len(p)):
                score[j] += f_arr[len(p),t+1]/(t+1)
                
    return score


def _g(n,t,p,g_arr):

    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n-1]*g_arr[n-1,t-1] + (1-p[n-1])*g_arr[n-1,t]


def _f(n,t,p,f_arr,g_arr):
    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n-1]*f_arr[n-1,t-1] + p[n-1]*t*g_arr[n-1,t-1]/n + (1-p[n-1])*f_arr[n-1,t]


# epistemic uncertainty:
def epistemic_uncertainty_pwc(clf, X_cand, precomp):
    freq = clf.predict_freq(X_cand)
    n = freq[:,0]
    p = freq[:,1]
    res = np.full((len(freq)), np.nan)
    if precomp is not None:
        # enlarges the precomp array if necessary:
        if precomp.shape[0] <= np.max(n)+1:
            new_shape = (int(np.max(n))-precomp.shape[0]+2, precomp.shape[1])
            precomp = np.append(precomp, np.full(new_shape, np.nan), axis=0)
        if precomp.shape[1] <= np.max(p)+1:
            new_shape = (precomp.shape[0], int(np.max(p))-precomp.shape[1]+2)
            precomp = np.append(precomp, np.full(new_shape, np.nan), axis=1)        

        for f in freq:
            # compute the epistemic uncertainty:
            for N in range(precomp.shape[0]):
                for P in range(precomp.shape[1]):
                    if np.isnan(precomp[N,P]):
                        pi1 = -minimize_scalar(_epistemic_pwc_sup_1,method='Bounded',bounds=(0.0,1.0), args=(N,P)).fun
                        pi0 = -minimize_scalar(_epistemic_pwc_sup_0,method='Bounded',bounds=(0.0,1.0), args=(N,P)).fun
                        pi = np.array([pi0,pi1])
                        precomp[N,P] = np.min(pi, axis=0)
        res = _interpolate(precomp,freq)
    else:
        for i, f in enumerate(freq):
            pi1 = -minimize_scalar(_epistemic_pwc_sup_1,method='Bounded',bounds=(0.0,1.0), args=(f[0],f[1])).fun
            pi0 = -minimize_scalar(_epistemic_pwc_sup_0,method='Bounded',bounds=(0.0,1.0), args=(f[0],f[1])).fun
            pi = np.array([pi0,pi1])
            res[i] = np.min(pi, axis=0)
    return res, precomp


def _interpolate(precomp, freq):
    # bilinear interpolation:
    points = np.zeros((precomp.shape[0]*precomp.shape[1],2))
    for n in range(precomp.shape[0]):
        for p in range(precomp.shape[1]):
            points[n*precomp.shape[1]+p] = n,p
    return griddata(points, precomp.flatten(), freq, method='linear')
    

def _epistemic_pwc_sup_1(t, n, p):
    if ((n == 0.0) and (p == 0.0)):
        return -1.0
    piH = ((t**p)*((1-t)**n))/(((p/(n+p))**p)*((n/(n+p))**n))
    return -np.minimum(piH,2*t-1)


def _epistemic_pwc_sup_0(t, n, p):
    if ((n == 0.0) and (p == 0.0)):
        return -1.0
    piH = ((t**p)*((1-t)**n))/(((p/(n+p))**p)*((n/(n+p))**n))
    return -np.minimum(piH,1-2*t)


#logistic regressionepistemic_uncertainty_logreg
#alg 3
def epistemic_uncertainty_logreg(X_cand, X, y, clf, probas):
    # compute pi0, pi1 for every x in X_cand:
    pi0, pi1 = np.empty((len(probas))), np.empty((len(probas)))
    for i, x in enumerate(X_cand):
        Qn = np.linspace(0.0,0.5,num=50, endpoint=False)
        Qp = np.linspace(0.5,1.0,num=50, endpoint=False)
        pi1[i], pi0[i] = np.maximum(2*probas[i]-1,0), np.maximum(1-2*probas[i],0)
        for q in range(100):
            idx_an, idx_ap = np.argmin(Qn), np.argmax(Qp)
            alpha_n, alpha_p = Qn[idx_an], Qp[idx_ap]
            if 2*ap-1 > pi1[i]:
                #solve 22 -> theta
                bounds = np.log(alpha_p/(1-alpha_p))
                A = np.insert(x,len(x),1)
                constraints = LinearConstraint(A=A, lb=bounds, ub=bounds)
                x0 = np.zeros((A.shape[0]))#
                theta = minimize(loglik_logreg, x0=x0, method='SLSQP', constraints=constraints, args=(clf,X,y)).x#
                pi1[i] = np.maximum(pi1[i],np.min(pi_h(theta, clf, X, y),2*ap-1))
            if 1-2*an > pi0[i]:
                #solve 22 -> theta
                pi0[i] = np.maximum(pi0[i],np.min(pi_h(theta, clf, X, y),1-2*an))

            Qn, Qp = np.delete(Qn, idx_an), np.delete(Qp, idx_ap)


    utilities = np.min(np.array([pi0,pi1]), axis=1)
    return utilities

def loglik_logreg(theta, clf, X, y, gamma=1):
    return _logistic_loss(theta, X, y, gamma, sample_weight=None)
    #L = np.exp(-_logistic_loss(theta, X, y, gamma, sample_weight=None))
    #
    #return clf.fit(X,y).get_parms(deep=False)#
    #
    c = theta[-1]
    theta = theta[:-1]
    lin = [c+np.dot(theta, x) for x in X]
    result = y.dot(lin)
    result -= np.sum(np.log(1+np.exp(lin)))
    result -= gamma*np.sum(theta**2)/2
    return -result

def pi_h(theta, clf, X, y, gamma=1):
    L_thata = -loglik_logreg(theta, clf, X, y, gamma)
    #can be precomputed:
    theta_ml = np.insert(clf.coef_,len(X), clf.intercept_, axis=0)
    L_ml = -loglik_logreg(theta_ml, clf, X, y, gamma=1)
    return L_theta/L_ml