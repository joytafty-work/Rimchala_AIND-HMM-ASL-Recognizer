import warnings
from asl_data import SinglesData
from collections import OrderedDict

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # TODO implement the recognizer
    probabilities = []
    guesses = []
    # Cast models into an ordered dictionary to ensure word ordering when getting guess_word
    models = OrderedDict(models)
    for (ix, (testing_X, testing_length)) in test_set.get_all_Xlengths().items():
        prob_dict = {}
        logLs = []
        default_logL = float("-inf")
        guess_word = ""
        # Find word in the training set that yield the best score given best model for that word
        for (word, model) in models.items():
            try:
                logL = model.score(testing_X, testing_length)
                prob_dict[word] = logL
                logLs.append(logL)
            except:
                prob_dict[word] = default_logL
                logLs.append(default_logL)

        guess_word = max(zip(models.keys(), logLs), key=lambda x: x[1])[0]

        probabilities.append(prob_dict)
        guesses.append(guess_word)

    return probabilities, guesses
