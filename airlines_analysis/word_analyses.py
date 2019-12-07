from wordcloud import WordCloud, STOPWORDS

class WordAnalysis:
    """
        Class that will hold all the analysis for words and things related to the analysis
        of the meaning of the tweets and data.
    """

    def get_word_cloud(self, df, sentiment, airline=None):
        df_final = df[df['sentiment'] == sentiment]

        if airline:
            df_final = df[df['airline'] == airline]


        words = words = ' '.join(df_final['text_final'])

        return WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(words)
