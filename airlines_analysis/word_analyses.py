from wordcloud import WordCloud,STOPWORDS

class WordAnalysis:
    """
        Class that will hold all the analysis for words and things related to the analysis
        of the meaning of the tweets and data.
    """

    def get_word_cloud(self, df, sentiment):
        df_final = df[df['sentiment'] == sentiment]

        words = words = ' '.join(df_final['text_final'])

        return WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(words)
