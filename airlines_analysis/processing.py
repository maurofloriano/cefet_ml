from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer


class Processing:
    """
        Class that will hold all the code for processing data, that includes 
        algorithms that we will run and validations.
    """

    @classmethod
    def tokenize(cls, text): 
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)
    
    @classmethod
    def get_count_vectorizer(cls):
        return CountVectorizer(
            analyzer = 'word',
            tokenizer = cls.tokenize,
            lowercase = True,
        )

