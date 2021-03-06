from airlines_analysis.word_analyses import WordAnalysis
from airlines_analysis.enums import SentimentEnum
import matplotlib.pyplot as plt

class ResultMixin(WordAnalysis):
    def get_airlines(self, df):
        airlines = df['airline']
        return list(set([a for a in airlines]))

    def word_cloud_results(self, df):
        airlines = self.get_airlines(df)
        nr_airlines = len(airlines)
        plt.figure(1, figsize=(20, 20))
        for index, airline in enumerate(airlines):
                word_cloud = self.get_word_cloud(df, SentimentEnum.NEGATIVE, airline)
                plt.subplot(nr_airlines / 3, 3, index + 1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(word_cloud)
                plt.title(airline)
        plt.show()

    def sentiment_results(self, df):
        airlines = self.get_airlines(df)
        nr_airlines = len(airlines)
        plt.figure(figsize=(20, 20))
        for index, airline in enumerate(airlines):
            filtred_df = df[df['airline'] == airline]
            count = filtred_df['airline_sentiment'].value_counts()
            plt.subplot(nr_airlines / 3, 3, index + 1)
            plt.bar([1, 2, 3], count)
            plt.xticks([1 ,2, 3], ['Negativo','Neutro','Positivo'])
            plt.ylabel('Número de tweets')
            plt.xlabel('Sentimento')
            plt.title(airline)
        plt.show()

    def all_sentiment_results(self, df):
        plt.figure(figsize=(20, 20))
        count = df['airline_sentiment'].value_counts()
        plt.bar([1, 2, 3], count)
        plt.xticks([1 ,2, 3], ['Negativo','Neutro','Positivo'])
        plt.ylabel('Número de tweets')
        plt.xlabel('Sentimento')
        plt.show()

    def all_word_cloud_results(self, df):
        plt.figure(1, figsize=(20, 20))
        word_cloud = self.get_word_cloud(df, SentimentEnum.NEGATIVE)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(word_cloud)
        plt.show()
