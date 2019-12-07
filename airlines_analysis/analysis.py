import nltk
import pandas as pd

from airlines_analysis.metrics import MetricsMixin
from airlines_analysis.preprocessing import PreprocessingMixin
from airlines_analysis.processing import Processing as ProcessingMixin
from airlines_analysis.results import ResultMixin

class Analysis(PreprocessingMixin, ProcessingMixin, MetricsMixin, ResultMixin):
    def __init__(self):
        nltk.download("stopwords")
        self.df = pd.read_csv("tweets.csv", encoding="ISO-8859-1")

    def run(self):
        self.preprocessing_df = self.preprocessing(self.df)
