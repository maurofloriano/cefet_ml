from airlines_analysis.analysis import Analysis
from airlines_analysis.enums import SentimentEnum
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

analysis = Analysis()
analysis.df
df_confindence_low = analysis.df[analysis.df["airline_sentiment:confidence"] < 0.60]
confident_df = analysis.remove_unconfident_sentiment(analysis.df)
web_cleaned_df = analysis.remove_html_and_tweet_tags(confident_df)
df_work = analysis.convert_sentiment_to_scalar(web_cleaned_df)
df_work = analysis.remove_stopwords(df_work)
df_work = df_work.loc[:, ['text_final', 'sentiment']]

# results = analysis.train(df_work)
# confusion_matrix = analysis.get_confusion_matrix(results)
# metrics_by_class = analysis.metrics_by_class(results)
# average_metrics = analysis.average_metrics(results)
results = analysis.cross_validation(df_work)
import pdb; pdb.set_trace()
# vect_data = tfidf.fit_transform(X_train)
# k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# clf = DecisionTreeClassifier()
# import pdb; pdb.set_trace()
# print(cross_val_score(clf, vect_data, y_test, cv=k_fold, n_jobs=1))
