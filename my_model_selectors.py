import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def bic_model(self, num_components):
        model = self.base_model(num_components)
        logL = model.score(self.X, self.lengths)
        num_features = model.n_features
        p = 2*num_features*num_components
        logN = np.log(len(self.X))
        bic = -2*logL + p*logN
        return bic, model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # Number of components = 2 * number of features * number of components 
        # factor of 2 coming from mean and variance

        models = []
        bics = []
        try:
            for nc in range(self.min_n_components, self.max_n_components+1):
                bic, model = self.bic_model(nc)
                models.append(model)
                bics.append(bic)
            best_model = max(zip(models, cv_logLs), key=lambda x: x[1])[0]
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def dic_model(self, num_components):
        model = self.base_model(num_components)
        logL = model.score(self.X, self.lengthz)
        return logL, model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        models = []
        logLs = []
        try:
            for nc in range(self.min_n_components, self.max_n_components+1):
                logL, model = self.dic_model(nc)
                models.append(model)
                logLs.append(logL)

            dics = [ll - np.mean(logLs[:ix] + logLs[ix:]) for (ix, ll) in enumerate(logLs)]
            best_model = max(zip(models, dics), key=lambda x: x[1])[0]
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def cv_model(self, num_components):
        """
        Calculate the average log likelihood of cross-validation folds using the KFold class
        :return: tuple of the mean likelihood and the model with the respective score
        """
        logLs = []
        split_method = KFold()

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)

            model = self.base_model(num_components)
            testing_X, testing_lengths = combine_sequences(cv_test_idx, self.sequences)
            logL = model.score(testing_X, testing_lengths)
            logLs.append(logL)
        return np.mean(logLs), model

    def select(self):
        """ select the best model for self.this_word based on
        CV score for n between self.min_n_components and self.max_n_components
        It is based on log likehood
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        cv_logLs = []
        try:
            for nc in range(self.min_n_components, self.max_n_components+1):
                logL, model = self.cv_model(nc)
                models.append(model)
                cv_logLs.append(logL)
            best_model = max(zip(models, cv_logLs), key=lambda x: x[1])[0]
            return best_model
        except:
            return self.base_model(self.n_constant)