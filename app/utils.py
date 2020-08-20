from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

class NamedEntityChecker(BaseEstimator, TransformerMixin):
    '''Custom Transformer that Searches for Named Entities'''

    def check_for_nnp(self, text):

        #Tokenize each word in the sentence, tag parts of speech
        pos_tagged = pos_tag(word_tokenize(text))

        #Extract all NNPs - Named Entities
        named_tokens = [item for item in pos_tagged if item[1]=='NNP']

        #If any named entity is listed, return True, otherwise False
        if len(named_tokens)>0:
            return True
        else:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply check_for_nnp function to all values in X
        X_tagged = pd.Series(X).apply(self.check_for_nnp)

        return pd.DataFrame(X_tagged)