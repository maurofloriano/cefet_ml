import re
from bs4 import BeautifulSoup
from airlines_analysis.enums import SentimentEnum

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


    def preprocessing(self, df):
        confident_df = self.remove_unconfident_sentiment(df)
        web_cleaned_df = self.remove_html_and_tweet_tags(confident_df)
        return self.convert_sentiment_to_scalar(web_cleaned_df)
