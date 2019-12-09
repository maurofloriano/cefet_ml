from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold


class Processing:
    """
        Class that will hold all the code for processing data, that includes
        algorithms that we will run and validations.
    """

    MODELS = [DecisionTreeClassifier, BernoulliNB, GaussianNB, AdaBoostClassifier]

    def tokenize(self, text):
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    def get_count_vectorizer(self):
        return CountVectorizer(analyzer="word", tokenizer=self.tokenize, lowercase=True)

    def get_train_test(self, df):
        train, test = train_test_split(df, test_size=0.2, random_state=1)
        X_train = train["text_final"].values
        X_test = test["text_final"].values
        y_train = train["sentiment"]
        y_test = test["sentiment"]
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
            results[c_model.__class__.__name__] = {"y_true": y_test, "y_pred": y_pred}

        return results

    def cross_validation(self, df):
        X_train, X_test, y_train, y_test = self.get_train_test(df)
        kf = KFold(n_splits=5, random_state=None)

        results = {}
        for class_model in self.MODELS:
            for index, (train_index, test_index) in enumerate(kf.split(X_train)):
                X_train_split, X_test_split = X_train[train_index], X_train[test_index]
                y_train_split, y_test_split = y_train.array[train_index], y_train.array[test_index]
                count_vectorizer = self.get_count_vectorizer()
                train_features = count_vectorizer.fit_transform(X_train_split)
                test_features = count_vectorizer.transform(X_test_split)
                c_model = class_model()
                c_model.fit(train_features.toarray(), y_train_split)
                y_pred = c_model.predict(test_features.toarray())
                if not results.get(index):
                    results[index] = {}
                results[index][c_model.__class__.__name__] = {"y_true": y_test_split, "y_pred": y_pred}

        return results
