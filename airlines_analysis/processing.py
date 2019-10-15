from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


class Processing:
    """
        Class that will hold all the code for processing data, that includes
        algorithms that we will run and validations.
    """
    MODELS = [
        DecisionTreeClassifier,
        BernoulliNB
    ]

    def tokenize(self, text):
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    def get_count_vectorizer(self):
        return CountVectorizer(
            analyzer='word',
            tokenizer=self.tokenize,
            lowercase=True,
        )

    def get_train_test(self, df):
        train, test = train_test_split(df, test_size=0.2, random_state=1)
        X_train = train['text_final'].values
        X_test = test['text_final'].values
        y_train = train['sentiment']
        y_test = test['sentiment']
        return X_train, X_test, y_train, y_test

    def train(self, df):
        X_train, X_test, y_train, y_test = self.get_train_test(df)
        count_vectorizer = self.get_count_vectorizer()
        train_features = count_vectorizer.fit_transform(X_train)
        test_features = count_vectorizer.transform(X_test)

        results = {}
        for class_model in self.MODELS:
            c_model = class_model()
            c_model.fit(train_features.toarray(), y_train)
            y_pred = c_model.predict(test_features.toarray())
            results[c_model.__class__.__name__] = {
                'y_true': y_test,
                'y_pred': y_pred,

            }

        return results
