"""
A Python implementation of the evidential k nearest neighbours for imperfectly labeled data.
EK-NN was first introduced by T. Denoeux and this version is based on a source code developed by Daniel Zhu.

Author : Arthur Hoarau
Date : 26/10/2021
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from ..utils import is_labeled
import numpy as np
import math

# Value of the Alpha parameter
ALPHA = 0.8
BETA = 2

class EKNN(BaseEstimator, ClassifierMixin):
    """
    EK-NN class used to predict labels when input data 
    are imperfectly labeled.
    
    Based on the Evidental k nearest neighbours (EKNN) classifier by Denoeux (1995).
    """

    def __init__(self, class_number, n_neighbors=5):
        """
        EK-NN class used to predict labels when input data 
        are imperfectly labeled.

        Parameters
        -----
        class_number: int
            The number of classes for the problem. Dimension of the possible classes.
        n_neighbors : int
            Number of nearest neighbors, default = 5

        Returns
        -----
        The instance of the class.
        """

        # Used to retrieve the n nearest neighbors
        self.n_neighbors = n_neighbors

        # Select number of classes
        self.nb_classes = 2**class_number - 1 
        self.class_number = class_number

        # Used to retrieve the state of the model
        self._fitted = False

    def get_params(self, deep=False):
        # Return the number of nearest neighbors as a dict
        return {"n_neighbors": self.n_neighbors, "class_number": self.class_number}

    def set_params(self, n_neighbors):
        # Set the number of nearest neighbors
        self.n_neighbors = n_neighbors

    def score(self, X, y_true, criterion=3):
        """
        Calculate the accuracy score of the model,
        unsig a specific criterion in "Max Credibility", 
        "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions.
        criterion : int
            Choosen criterion for prediction, by default criterion = 3.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        y_pred = self.predict(X, criterion=criterion)

        # Compare with true labels, and compute accuracy
        return accuracy_score(y_true, y_pred)
    
    def partial_fit(self, X, y):
        self.fit(X, y)

    def fit(self, X, y, alpha=ALPHA, beta=BETA, unique_gamma=True):
        """
        Fit the model according to the training data.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y : ndarray
            Labels array
        alpha : int
            Value of the alpha parameter, default = 0.95
        beta : int
            Value of the beta parameter, default = 1.5
        unique_gamma : boolean
            True for a unique computation of a global gamma parameter, 
            False for multiple gammas (high computational cost). default = True.
        Returns
        -----
        self : EKNN
            The instance of the class.
        """

        is_lbld = is_labeled(y, missing_label="nan")
        X = X[is_lbld]
        y = y[is_lbld]
        
        # Cast y into evidential
        if (y.ndim == 1):
            y_cred = np.zeros((y.shape[0], self.nb_classes + 1))
            for i in range(0, y.shape[0]):
                y_cred[i][2**int(y[i])] = 1
            y = y_cred

        # Check for data integrity
        if X.shape[0] != y.shape[0]:
            if X.shape[0] * (self.nb_classes + 1) == y.shape[0]:
                y = np.reshape(y, (-1, self.nb_classes + 1))
            else:
                raise ValueError("X and y must have the same number of rows")
            
        if X.shape[0] < self.n_neighbors:
            raise ValueError("Not enough data to match n_neighbors:", self.n_neighbors)


        # Verify if the size of y is of a power set (and if it contains the empty set or not)
        if math.log(y.shape[1], 2).is_integer():
            y = y[:,1:]
        elif not math.log(y.shape[1] + 1, 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")

        # Save X and y
        self.X_trained = X
        self.y_trained = y

        # Save size of the dataset
        self.size = self.X_trained.shape[0]

        # Init gamma and alpha
        self._init_parameters(alpha=alpha, unique_gamma=unique_gamma, beta=beta)

        # The model is now fitted
        self._fitted = True

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled

        Returns
        -----
        predictions : ndarray
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        result = self._predict(X)

        predictions = EKNN.decisionDST(result.T, 4, return_prob=True)

        return predictions


    def predict(self, X, criterion=3, return_bba=False):
        """
        Predict labels of input data. Can return all bbas. Criterion are :
        "Max Credibility", "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled
        creterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".
        return_bba : boolean
            Type of return, predictions or both predictions and bbas, 
            by default return_bba=False.

        Returns
        -----
        predictions : ndarray
        result : ndarray
            Predictions if return_bba is False and both predictions and masses if return_bba is True
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        # Predict output bbas for X
        result = self._predict(X)

        # Max Plausibility
        if criterion == 1:
            predictions = EKNN.decisionDST(result.T, 1)
        # Max Credibility
        elif criterion == 2:
            predictions = EKNN.decisionDST(result.T, 2)
        # Max Pignistic probability
        elif criterion == 3:
            predictions = EKNN.decisionDST(result.T, 4)
        else:
            raise ValueError("Unknown decision criterion")

        # Return predictions or both predictions and bbas
        if return_bba:
            return predictions, result
        else:
            return predictions

    def _compute_bba(self, X, indices, distances):
        """
        Compute the bba for each element of X.

        Parameters
        -----
        X : ndarray
            Input array of X
        indices : ndarray
            Array of K nearest neighbors indices
        distances : ndarray
            Array of K nearest neighbors distances

        Returns
        -----
        bba : ndarray
            Array of bbas
        """
        # Initialisation of size and all bba
        n_samples = X.shape[0]
        bba = np.zeros((n_samples, self.nb_classes + 1))

        # Calculate a bba for each element of X
        for i in range(n_samples):
            m_list = np.zeros((self.n_neighbors, self.nb_classes + 1))

            # Construct a bba for each neighbors
            for j in range(self.n_neighbors):
                m = np.zeros(self.nb_classes + 1)
                m[-1] = 1

                for c in range(m.shape[0] - 2):
                    if isinstance(self.gamma, float):
                        weight = self.alpha * math.exp((-self.gamma) * (distances[i,j] ** self.beta)) * self.y_trained[int(indices[i,j]), c]
                    else:
                        weight = self.alpha * math.exp((-self.gamma[int(indices[i,j])]) * (distances[i,j] ** self.beta)) * self.y_trained[int(indices[i,j]), c]
                    m[c + 1] = weight
                    m[-1] -= weight

                m_list[j] = m
            
            # Compute normalized combination of bba
            m_normalized = np.array(EKNN.DST(m_list.T, 2))

            # Append the normalized bba to the array
            bba[i] = m_normalized.T
        
        return bba

    def _compute_distances(self, X):
        """
        Compute the euclidian distances with each neighbors.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        indices : ndarray
            Array of K nearest neighbors indices
        distances : ndarray
            Array of K nearest neighbors distances
        """

        # Initialize indices and nearest neighbors and distances
        indices = np.zeros((X.shape[0], self.n_neighbors))
        distances = np.zeros((X.shape[0], self.n_neighbors))

        # Loop over every input sample
        for i in range(X.shape[0]):

            # Compute the distance (without sqrt)
            dist = np.sqrt(np.sum(([X[i]] - self.X_trained)**2,axis=1))
            sorted_indices = np.argsort(dist)[:self.n_neighbors]

            # Append result to each arrays
            indices[i] = sorted_indices
            distances[i] = dist[sorted_indices]
        return indices, distances

    def _predict(self, X):
        """
        Compute distances and predicted bba on the input.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        result : ndarray
            Array of normalized bba
        """
        
        # Compute distances with k nearest neighbors
        neighbours_indices, neighbours_distances = self._compute_distances(X)

        # Compute bba
        result = self._compute_bba(X, neighbours_indices, neighbours_distances)

        return result

    def _init_parameters(self, alpha=ALPHA, beta=BETA, unique_gamma=False):
        # Init alpha and beta
        self.alpha = alpha
        self.beta = beta

        # Init parameter gamma
        self.gamma = self._compute_gamma(unique_gamma=unique_gamma)

    def _compute_gamma(self, unique_gamma=False):
        """
        Compute gamma parameter. Either unique or multiple.

        Returns
        -----
        gamma : ndarray
            Array of gamma parameters
        or
        gamma : int
            Value of gamma
        """

        if(unique_gamma):
            # Initialize distances and divider term
            divider = (self.size**2 - self.size) if self.size > 1 else 1
            distances = np.zeros((self.size, self.size))

            # Compute euclidian distances between each point
            for i, x in enumerate(self.X_trained):
                distances[i] = np.sqrt(
                    np.sum(([x] - self.X_trained)**2,axis=1)
                )

            mean_distance = np.sum(distances) / divider
            return 1 / (mean_distance  ** self.beta)

        # Initialize distances and divider term
        gamma = np.zeros(self.size)

        jousselme_distance = np.zeros((self.size, self.size))
        norm_distances = np.zeros((self.size, self.size))

        # Compute Jousselme and norm distances
        for i in range(self.size):

            # Init masses
            bbai = np.zeros(self.nb_classes + 1)
            bbai[1:] = self.y_trained[i]

            D = EKNN.Dcalculus(np.array(bbai).reshape((1,bbai.size)).size)

            for j in range(self.size):
                
                # Init masses
                bbaj = np.zeros(self.nb_classes + 1)
                bbaj[1:] = self.y_trained[j]

                jousselme_distance[i, j] = EKNN.JousselmeDistance(bbai, bbaj, D)

        norm_distances = np.array([[np.linalg.norm(i-j) for j in self.X_trained] for i in self.X_trained])

        for n in range(self.size):

            # Init the bba
            bban = np.zeros(self.nb_classes + 1)
            bban[1:] = self.y_trained[n]

            jousselm_distances_matrix = np.zeros((1, self.size))
            jousselm_distances_matrix[0] =  1 - jousselme_distance[n]

            jousselm_product = np.matmul(jousselm_distances_matrix.T, jousselm_distances_matrix)

            # Buffer not to compute multiple times the operation
            dividend = jousselm_product * norm_distances
            divisor = np.sum(jousselm_product) - np.sum(np.diagonal(jousselm_product))

            # If Jousselme distances are nulls
            if divisor == 0:
                divisor = 1

            gamma[n] = 1 / ((np.sum(dividend) / divisor) ** self.beta)
        return gamma
    
    @staticmethod
    def decisionDST(mass, criterion, r=0.5, return_prob=False):
        """Different rules for decision making in the framework of belief functions
        
        Parameters
        -----------
        mass: mass function to decide with
        
        criterion: integer
            different decision rules to apply.
                criterion=1 maximum of the plausibility
                criterion=2 maximum of the credibility
                criterion=3 maximum of the credibility with rejection
                criterion=4 maximum of the pignistic probability
                criterion=5 Appriou criterion (decision onto \eqn{2^\Theta})
        """
        mass = mass.copy()
        if (mass.size in mass.shape):   #if mass is a 1*N or N*1 array
            mass = mass.reshape(mass.size,1)
        nbEF, nbvec_test = mass.shape   #number of focal elements = number of mass rows 
        nbClasses = round(math.log(nbEF,2))
        class_fusion = []


        for k in range(nbvec_test):
            massTemp = mass[:,k]

            #Select only singletons
            natoms = round(math.log(massTemp.size, 2))
            singletons_indexes = np.zeros(natoms)
            for i in range(natoms):
                if i == 1:
                    singletons_indexes[i] = 2
                else:
                    singletons_indexes[i] = 2**i

            singletons_indexes = singletons_indexes.astype(int)

            if criterion == 1:
                pl = np.array(EKNN.mtopl(massTemp))
                indice = np.argmax(pl[singletons_indexes])
                class_fusion.append(indice)
            elif criterion == 2:
                bel = np.array(EKNN.mtobel(massTemp))
                indice = np.argmax(bel[singletons_indexes])
                class_fusion.append(indice)
            elif criterion == 4:
                pign = np.array(EKNN.mtobetp(massTemp.T))
                if return_prob:
                    indice = pign
                else:
                    indice = np.random.choice(np.flatnonzero(pign == pign.max()))
                class_fusion.append(indice)

        return np.array(class_fusion)

    @staticmethod
    def mtonm(InputVec):
        """
        Transform bbm into normalized bbm
        Parameter
        ---------
        InputVec: vector m representing a mass function
        Return
        ---------
        out: vector representing a normalized mass function
        """
        if InputVec[0] < 1:
            out = InputVec/(1-InputVec[0])
            out[0] = 0
        return out

    @staticmethod
    def mtobel(InputVec):
        return EKNN.mtob(EKNN.mtonm(InputVec))

    @staticmethod
    def mtob(InputVec):
        """
        Comput InputVec from m to b function.  belief function + m(emptset)
        Parameter
        ---------
        InputVec: vector m representing a mass function
        Return
        ---------
        out: a vector representing a belief function
        """
        InputVec = InputVec.copy()
        mf = InputVec.size
        natoms = round(math.log(mf, 2))
        if math.pow(2, natoms) == mf:
            for i in range(natoms):
                i124 = int(math.pow(2, i))
                i842 = int(math.pow(2, natoms - i))
                i421 = int(math.pow(2, natoms - i - 1))
                InputVec = InputVec.reshape(i124, i842, order='F')
                # for j in range(1, i421 + 1): #to be replaced by i842
                for j in range(i421):  # not a good way for add operation coz loop matrix for i842 times
                    InputVec[:, j * 2 + 1] = InputVec[:,
                        j * 2 + 1] + InputVec[:, j * 2]
            out = InputVec.reshape(1, mf, order='F')[0]
            return out
        else:
            raise ValueError(
                "ACCIDENT in mtoq: length of input vector not OK: should be a power of 2, given %d\n" % mf)

    @staticmethod
    def btopl(InputVec):
        """
        Compute from belief b to plausibility pl
        Parameter
        ---------
        InputVec: belief function b
        Return
        ------
        out: plausibility function pl
        """

        lm = InputVec.size
        natoms = round(math.log2(lm))
        if 2 ** natoms == lm:
            InputVec = InputVec[-1] - InputVec[::-1]
            out = InputVec
            return out
        else:
            raise ValueError("ACCIDENT in btopl: length of input vector not OK: should be a power of 2, given %d\n" % lm)


    @staticmethod
    def mtopl(InputVec):
        """
        Compute from mass function m to plausibility pl.
        Parameter
        ----------
        InputVec: mass function m
        Return
        --------
        output: plausibility function pl
        """
        InputVec = EKNN.mtob(InputVec)
        out = EKNN.btopl(InputVec)
        return out

    @staticmethod
    def mtoq(InputVec):
        """
        Computing Fast Mobius Transfer (FMT) from mass function m to commonality function q
        Parameters
        ----------
        InputVec : vector m representing a mass function
        Return:
        out: a vector representing a commonality function
        """
        InputVec = InputVec.copy()
        mf = InputVec.size
        natoms = round(math.log2(mf))
        if 2 ** natoms == mf:
            for i in range(natoms):
                i124 = int(math.pow(2, i))
                i842 = int(math.pow(2, natoms - i))
                i421 = int(math.pow(2, natoms - i - 1))
                InputVec = InputVec.reshape(i124, i842, order='F')
                # for j in range(1, i421 + 1): #to be replaced by i842
                for j in range(i421):  # not a good way for add operation coz loop matrix for i842 times
                    InputVec[:, j * 2] = InputVec[:, j * 2] + InputVec[:, j * 2+1]
            out = InputVec.reshape(1, mf, order='F')[0]
            return out
        else:
            raise ValueError(
                "ACCIDENT in mtoq: length of input vector not OK: should be a power of 2, given %d\n" % mf)
    
    @staticmethod
    def qtom(InputVec):
        """
        Compute FMT from q to m.
        Parameter
        ----------
        InputVec: commonality function q
        Return
        --------
        output: mass function m
        """
        InputVec = InputVec.copy()
        lm = InputVec.size
        natoms = round(math.log(lm, 2))
        if math.pow(2, natoms) == lm:
            for i in range(natoms):
                i124 = int(math.pow(2, i))
                i842 = int(math.pow(2, natoms - i))
                i421 = int(math.pow(2, natoms - i - 1))
                InputVec = InputVec.reshape(i124, i842, order='F')
                # for j in range(1, i421 + 1): #to be replaced by i842
                for j in range(i421):  # not a good way for add operation coz loop matrix for i842 times
                    InputVec[:, j * 2] = InputVec[:, j * 2] - InputVec[:, j * 2+1]
            out = InputVec.reshape(1, lm, order='F')[0]
            return out
        else:
            raise ValueError("ACCIDENT in qtom: length of input vector not OK: should be a power of 2\n")

    @staticmethod
    def Dcalculus(lm):
        """Compute the Jaccard matrix for the disernment framework of the given mass function
        Parameter
        ---------
        lm: a vector representing a mass function
        Return
        ------
        out: the Jaccard matrix for the given mass function
        """
        natoms = round(math.log2(lm))
        ind = [{}]*lm
        if (math.pow(2, natoms) == lm):
            ind[0] = {0} #In fact, the first element should be a None value (for empty set).
            #But in the following calculate, we'll deal with 0/0 which shoud be 1 bet in fact not calculable. So we "cheat" here to make empty = {0}
            ind[1] = {1}
            step = 2
            while (step < lm):
                ind[step] = {step}
                step = step+1
                indatom = step
                for step2 in range(1,indatom - 1):
                    #print(type(ind[step2]))
                    ind[step] = (ind[step2] | ind[indatom-1])
                    #ind[step].sort()
                    step = step+1
        out = np.zeros((lm,lm))

        for i in range(lm):
            for j in range(lm):
                out[i][j] = float(len(ind[i] & ind[j]))/float(len(ind[i] | ind[j]))
        return out

    @staticmethod
    def JousselmeDistance(mass1, mass2, D = "None"):
        """
        Calclate Jousselme distance between two mass functions mass1 and mass2
        This function is able to calcuate the Jaccard matrix if not given. 
        Attention: calculation of Jaccard matrix is a heavy task.
        Parameters
        ----------
        mass1: a vector representing first mass function
        mass2: a vector representing other mass function

        D: a square matrix representing Jaccard matrix

        Return
        ------
        out: float representing the distance
        """
        m1 = np.array(mass1).reshape((1,mass1.size))
        m2 = np.array(mass2)
        if m1.size != m2.size:
            raise ValueError("mass vector should have the same size, given %d and %d" % (m1.size, m2.size))
        else:
            if type(D)== str:
                D = EKNN.Dcalculus(m1.size)
            m_diff = m1 - m2
            out = math.sqrt(np.dot(np.dot(m_diff,D),m_diff.T)/2.0)
            return out

    @staticmethod
    def mtobetp(InputVec):
        """Computing pignistic propability BetP on the signal points from the m vector (InputVec) out = BetP
        vector beware: not optimize, so can be slow for more than 10 atoms
        Parameter
        ---------
        InputVec: a vector representing a mass function
        Return
        ---------
        out: a vector representing the correspondant pignistic propability
        """
        # the length of the power set, f
        mf = InputVec.size
        # the number of the signal point clusters
        natoms = round(math.log(mf, 2))
        if math.pow(2, natoms) == mf:
            if InputVec[0] == 1:
                # bba of the empty set is 1
                raise ValueError("warning: all bba is given to the empty set, check the frame\n")
            else:
                betp = np.zeros(natoms)
                for i in range(1, mf):
                    # x , the focal sets InputVec the dec2bin form
                    x = np.array(list(map(int, np.binary_repr(i, natoms)))[
                                ::-1])  # reverse the binary expression
                    # m_i is assigned to all the signal points equally

                    betp = betp + np.multiply(InputVec[i]/sum(x), x)
                out = np.divide(betp, (1.0 - InputVec[0]))
            return out
        else:
            raise ValueError(
                "Error: the length of the InputVec vector should be power set of 2, given %d \n" % mf)

    @staticmethod 
    def DST(massIn, criterion, TypeSSF=0):
        """
        Combination rules for multiple masses.
        Parameters
        ----------
        massIn: ndarray
            Masses to be combined, represented by a 2D matrix
        criterion: integer
            Combination rule to be applied
            The criterion values represented respectively the following rules:
                criterion=2 Dempster-Shafer criterion (normalized)
        Return
        ----------
        Mass: ndarray
            a final mass vector combining all masses
        """
        n, m = massIn.shape
        if criterion in (1,2,3,6,7, 14):

            q_mat = np.apply_along_axis(EKNN.mtoq, axis = 0, arr = massIn) #apply on column. (2 in R)
            q = np.apply_along_axis(np.prod, axis = 1,arr = q_mat) # apply on row (1 in R)
        if criterion == 2:
            #Dempster-Shafer criterion (normalized)
            Mass = EKNN.qtom(q)
            Mass = Mass/(1-Mass[0])
            Mass[0] = 0
        
        return Mass[np.newaxis].transpose()

