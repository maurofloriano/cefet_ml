import pandas as pd
from airlines_analysis.preprocessing import PreprocessingMixin

class Analysis(PreprocessingMixin):
    def __init__(self):
        self.df = pd.read_csv('tweets.csv', encoding='ISO-8859-1')

    def run(self):
        self.preprocessing_df = self.preprocessing(self.df)
