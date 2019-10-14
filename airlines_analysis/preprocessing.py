import re
from bs4 import BeautifulSoup
from airlines_analysis.enums import SentimentEnum
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from nltk.corpus import stopwords


class PreprocessingMixin:
    def remove_unconfident_sentiment(self, df):
        return df[df["airline_sentiment:confidence"] >= 0.60]

    def remove_html_and_tweet_tags(self, df):
        def cleaner_function(data):
            html_and_tweet_cleaner_functions = [
                lambda x: BeautifulSoup(x, 'lxml').text,
                lambda x: re.sub( r'@[\w]*', '', str(x)),
                lambda x: re.sub( r'http\S+', '', str(x))
            ]
            cleaned_data = data
            for cleaner_function in html_and_tweet_cleaner_functions:
                cleaned_data = cleaner_function(cleaned_data)
            return cleaned_data
        cleaned_df = df
        cleaned_df.insert(2, 'text_final', df['text'].apply(cleaner_function))
        return cleaned_df

    def convert_sentiment_to_scalar(self, df):
        def sentiment_to_scalar(data):
            if data == 'neutral':
                return SentimentEnum.NEUTRO
            if data == 'negative':
                return SentimentEnum.NEGATIVE
            if data == 'positive':
                return SentimentEnum.POSITIVE
            return SentimentEnum.UNKNOWN

        converted_df = df
        converted_df.insert(
            2, 'sentiment', df['airline_sentiment'].apply(sentiment_to_scalar))
        return converted_df


    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words)



    def preprocessing(self, df):
        confident_df = self.remove_unconfident_sentiment(df)
        web_cleaned_df = self.remove_html_and_tweet_tags(confident_df)
        web_cleaned_df.text_final = web_cleaned_df.text_final.apply(remove_stopwords)
        return self.convert_sentiment_to_scalar(web_cleaned_df)
    
    def get_train_test(self, df):
        train, test = train_test_split(df, test_size=0.2, random_state=1)
        X_train = train['text_final'].values
        X_test = test['text_final'].values
        y_train = train['sentiment']
        y_test = test['sentiment']
        return X_train, X_test, y_train, y_test
