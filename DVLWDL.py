import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from pyts.approximation import DiscreteFourierTransform, SymbolicFourierApproximation
from pyts.base import UnivariateTransformerMixin
from pyts.utils.utils import _windowed_view, windowed_view
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from math import ceil


class DVLWDL(BaseEstimator, UnivariateTransformerMixin):
    def __init__(self, word_size=4, n_bins=4,
                 window_sizes=[70, 20, 30], window_steps=[1, 1, 1],
                 anova=True, drop_sum=True, norm_mean=True, norm_std=True,
                 strategy='entropy', chi2_threshold=2, sparse=True,
                 alphabet=None, skip = 1, theta = 3,minF=4,maxF=10,delta=2):
        self.word_size = word_size
        self.n_bins = n_bins
        self.window_sizes = window_sizes
        self.window_steps = window_steps
        self.anova = anova
        self.drop_sum = drop_sum
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.strategy = strategy
        self.chi2_threshold = chi2_threshold
        self.sparse = sparse
        self.alphabet = alphabet
        self.skip = skip # skip gram parameter
        self.theta = theta
        self.minF=minF
        self.maxF=maxF
        self.delta=delta
    
    def computeWordsLength(self, X, y):
        wordLength=self.minF
        dft = DiscreteFourierTransform(n_coefs=self.maxF,anova=self.anova,norm_mean=self.norm_mean,
                                    norm_std=self.norm_std,drop_sum=self.drop_sum)
        A = dft.fit_transform(X,y)
        Fstat=f_classif(A,y)[0] ## compute the F statistic
        p1=np.sum(Fstat[0:self.minF])
        p2=np.sum(Fstat[self.minF:])

        if p1>p2:
            return wordLength
        else:
            p=p1+(p2-p1)/self.delta
            while p1<p and wordLength<=self.maxF:
                p1=p1+Fstat[wordLength]+Fstat[wordLength+1]
                wordLength+=2
            return wordLength

    def generate_wordLength(self, X,y):
        n_samples, n_timestamps = X.shape
        window_sizes = self.window_sizes
        window_steps = self.window_steps
       
        wordLengths={}

        for (window_size, window_step) in zip(window_sizes, window_steps):

            n_windows = ((n_timestamps - window_size + window_step) // window_step)

            X_windowed = windowed_view(X, window_size=window_size,window_step=window_step)
            X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)
            y_repeated = np.repeat(y, n_windows)

            wordLength = self.computeWordsLength(X=X_windowed,y=y_repeated)
            wordLengths[window_size]= wordLength
        return wordLengths

    def g(self, accuracies, theta ):
        acc_max = max(accuracies)
        return theta * ( 0.01*(acc_max - accuracies <= 0.15)  +
                        0.2*np.logical_and( acc_max - accuracies <= 0.25, acc_max - accuracies > 0.15 )+
                        0.3*np.logical_and( acc_max - accuracies <= 0.35, acc_max - accuracies > 0.25) +
                        0.4*np.logical_and( acc_max - accuracies <= 0.45, acc_max - accuracies > 0.35) +
                        0.5*( acc_max - accuracies > 0.45)
        )
    
    def createTfIdfWordDictionary(self, wordList, thresholds , X, y):
        
        # dictionary: word -> (class, index of the window)
        frequencies = {} # For each class, contains the list of frequencies of each word
        TfIdfWordDict = set() # Contain final words
        
        X_word_list = self.X_word_list #  contains the list of words for each training example,

        for i in range(len(wordList)):
            word, c, idx = wordList[i]
            freq_t_c = sum([ 1/len(X_word_list[i]) * X_word_list[i].count(word) for i in range(len(X_word_list)) if y[i] == c])
            frequencies[c] = frequencies.get(c, []) + [freq_t_c] 
            
        for i in range(len(wordList)):
            word, c, idx = wordList[i] # c is the class, idx is the index of the window
            freq_t_c = sum([ 1/len(X_word_list[i]) * X_word_list[i].count(word) for i in range(len(X_word_list)) if y[i] == c])
            tf = np.log(freq_t_c / max(frequencies[c])+1) 

            N_c_bar = sum([1 for i in range(len(X_word_list)) if y[i] != c])
            inverse_frequency_word_c = sum([1 for i in range(len(X_word_list)) if word in X_word_list[i] and y[i] != c])
            idf = np.log(N_c_bar / (inverse_frequency_word_c+1))
            tf_idf = tf * idf
            if tf_idf > thresholds[idx]: # Problem : a word can be in several windows
                TfIdfWordDict.add(word)
        return TfIdfWordDict

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        self : object

        """
        X, y = check_X_y(X, y, dtype='float64')
        check_classification_targets(y)

        n_samples, n_timestamps = X.shape
        window_sizes, window_steps = self.window_sizes,self.window_steps
        
        self._sfa_list = []
        self.WordList = [] # Word list
        self.TfIdfWordDict = {} # Tf-idf Word Dictionary
        # LEARNING THE WORD LENGTH FOR EACH SLIDING WINDOW
        self.wordLengths = self.generate_wordLength(X,y)
        self.accuracies = []
        self.X_word_list = [[] for _ in range(len(X))] # List of words for each time series
        idx = 0
        for (window_size, window_step) in zip(window_sizes, window_steps):
            word_size_window = self.wordLengths[window_size]
            n_windows = ((n_timestamps - window_size + window_step)
                         // window_step)
            X_windowed = windowed_view(X, window_size, window_step)
            
            X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)
            sfa = SymbolicFourierApproximation(
                n_coefs=word_size_window, drop_sum=self.drop_sum,
                anova=self.anova, norm_mean=self.norm_mean,
                norm_std=self.norm_std, n_bins=self.n_bins,
                strategy=self.strategy, alphabet=self.alphabet
            )

            y_repeated = np.repeat(y, n_windows)
            X_sfa = sfa.fit_transform(X_windowed, y_repeated)
            X_word = np.asarray([''.join(X_sfa[i])
                                 for i in range(n_samples * n_windows)])
            
            X_word = X_word.reshape(n_samples, n_windows)
            X_word_copy=[]
            #ADDING The size of the window before the word
            for i in range(len(X_word)):
                X_word_copy.append([])
                for j in range(len(X_word[i])):
                    X_word_copy[i].append(str(window_size) + X_word[i][j])

            X_word=X_word_copy

            for i in range(len(X)):
                self.X_word_list[i].extend(list(X_word[i]))
            # BUILDING THE DISCRIMINATIVE WORD DICTIONARY
            # We add the unigrams and the skip-grams with classes, for tf-idf later
            
            for i, words in enumerate(X_word):
                for j in range(len(words)):
                    # Adding the unigrams
                    self.WordList.append((words[j], y[i], idx)) # We store the word, the class and the index of the window
                    # Adding the skipgrams, taking non-overlapping windows
                    first_skip_gram_idx = max(j- self.skip * ceil(window_size/window_step) - 1, 0)
                    for k in range(j-1, first_skip_gram_idx, -ceil(window_size/window_step)):
                        skip_gram_word = '&'.join([words[k], words[j]])
                        self.X_word_list[i].append(skip_gram_word)
                        self.WordList.append((skip_gram_word, y[i], idx))

            # CREATING TfIdf WordDictionary

            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
            
            vectorizer = CountVectorizer(ngram_range=(1, 2))
            X_counts = vectorizer.fit_transform(X_bow)
            
            lr = LogisticRegression(solver = 'liblinear')
            self.accuracies.append(cross_val_score(lr, X_counts, y, cv = 10).mean())
            idx += 1
            self._sfa_list.append(sfa)


        self.accuracies = np.asarray(self.accuracies)
        self.thresholds = self.g(self.accuracies, self.theta)

        # Computing the tf-idf for each word, class
        self.TfIdfWordDict = self.createTfIdfWordDictionary(self.WordList,self.thresholds, X, y )
        

        # Creating frequency vector representation of the time series using the TfIdfWordDict

        X_bow = np.asarray([' '.join(self.X_word_list[i]) for i in range(n_samples)])
        vectorizer = CountVectorizer(vocabulary=self.TfIdfWordDict)
        X_counts = vectorizer.fit_transform(X_bow)
        self.last_vectorizer = vectorizer
        self.X_counts = X_counts

        return self


    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X_new : sparse matrix, shape = (n_samples, n_features)
            Document-term matrix with relevant features only.

        """
        check_is_fitted(self, ['_sfa_list', 'TfIdfWordDict', 'last_vectorizer'])

        X = check_array(X, dtype='float64')

        n_samples, n_timestamps = X.shape

        X_features = coo_matrix((n_samples, 0), dtype=np.int64)
        
        X_word_list = [[] for _ in range(len(X))] # List of words for each time series
        for (window_size, window_step, sfa) in zip(
                 self.window_sizes, self.window_steps, self._sfa_list):

            n_windows = ((n_timestamps - window_size + window_step)
                         // window_step)
            X_windowed = windowed_view(X, window_size, window_step)
            X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)
            X_sfa = sfa.transform(X_windowed)

            X_word = np.asarray([''.join(X_sfa[i])
                                 for i in range(n_samples * n_windows)])
            X_word = X_word.reshape(n_samples, n_windows)
            X_word_copy=[]
            #ADDING The size of the window before the word
            for i in range(len(X_word)):
                X_word_copy.append([])
                for j in range(len(X_word[i])):
                    X_word_copy[i].append(str(window_size) + X_word[i][j])

            X_word=X_word_copy
            for i in range(len(X)):
                X_word_list[i].extend(list(X_word[i]))

            for i, words in enumerate(X_word):
                for j in range(len(words)):
                    # Adding the skipgrams, taking non-overlapping windows
                    for k in range(0, j-1,ceil(window_size/window_step)):
                        skip_gram_word = '&'.join([words[k], words[j]])
                        X_word_list[i].append(skip_gram_word)
        
            
        X_bow = np.asarray([' '.join(X_word_list[i]) for i in range(n_samples)])
        X_features = self.last_vectorizer.transform(X_bow)

        if not self.sparse:
            return X_features.A
        return csr_matrix(X_features)


    def fit_transform(self, X, y):
        """Fit the data then transform it.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Train samples.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        X_new : array, shape (n_samples, n_words)
            Document-term matrix.

        """
        X, y = check_X_y(X, y, dtype='float64')
        check_classification_targets(y)
        n_samples, n_timestamps = X.shape
        window_sizes, window_steps = self.window_sizes,self.window_steps
        
        self._sfa_list = []
        self.WordList = [] # Word list
        self.TfIdfWordDict = {} # Tf-idf Word Dictionary
        # LEARNING THE WORD LENGTH FOR EACH SLIDING WINDOW
        self.wordLengths = self.generate_wordLength(X,y)
        self.accuracies = []
        self.X_word_list = [[] for _ in range(len(X))] # List of words for each time series
        idx = 0
        for (window_size, window_step) in zip(window_sizes, window_steps):
            word_size_window = self.wordLengths[window_size]
            n_windows = ((n_timestamps - window_size + window_step)
                         // window_step)
            X_windowed = windowed_view(X, window_size, window_step)
            
            X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)
            sfa = SymbolicFourierApproximation(
                n_coefs=word_size_window, drop_sum=self.drop_sum,
                anova=self.anova, norm_mean=self.norm_mean,
                norm_std=self.norm_std, n_bins=self.n_bins,
                strategy=self.strategy, alphabet=self.alphabet
            )

            y_repeated = np.repeat(y, n_windows)
            X_sfa = sfa.fit_transform(X_windowed, y_repeated)
            X_word = np.asarray([''.join(X_sfa[i])
                                 for i in range(n_samples * n_windows)])
            
            X_word = X_word.reshape(n_samples, n_windows)
            X_word_copy=[]
            #ADDING The size of the window before the word
            for i in range(len(X_word)):
                X_word_copy.append([])
                for j in range(len(X_word[i])):
                    X_word_copy[i].append(str(window_size) + X_word[i][j])

            X_word=X_word_copy

            for i in range(len(X)):
                self.X_word_list[i].extend(list(X_word[i]))
            # BUILDING THE DISCRIMINATIVE WORD DICTIONARY
            # We add the unigrams and the skip-grams with classes, for tf-idf later
            
            for i, words in enumerate(X_word):
                for j in range(len(words)):
                    # Adding the unigrams
                    self.WordList.append((words[j], y[i], idx)) # We store the word, the class and the index of the window
                    # Adding the skipgrams, taking non-overlapping windows
                    for k in range(0, j-1,ceil(window_size/window_step)):
                        skip_gram_word = '&'.join([words[k], words[j]])
                        self.X_word_list[i].append(skip_gram_word)
                        self.WordList.append((skip_gram_word, y[i], idx))

            # CREATING TfIdf WordDictionary

            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
            
            vectorizer = CountVectorizer(ngram_range=(1, 2))
            X_counts = vectorizer.fit_transform(X_bow)
            
            lr = LogisticRegression(solver = 'liblinear')
            self.accuracies.append(cross_val_score(lr, X_counts, y, cv = 10).mean())
            idx += 1
            self._sfa_list.append(sfa)


        self.accuracies = np.asarray(self.accuracies)
        self.thresholds = self.g(self.accuracies, self.theta)

        # Computing the tf-idf for each word, class
        self.TfIdfWordDict = self.createTfIdfWordDictionary(self.WordList,self.thresholds, X, y )
        

        # Creating frequency vector representation of the time series using the TfIdfWordDict

        X_bow = np.asarray([' '.join(self.X_word_list[i]) for i in range(n_samples)])
        vectorizer = CountVectorizer(vocabulary=self.TfIdfWordDict)
        X_counts = vectorizer.fit_transform(X_bow)
        self.last_vectorizer = vectorizer
        X_features = self.last_vectorizer.transform(X_bow)


        if not self.sparse:
            return X_features.A
        return csr_matrix(X_features)