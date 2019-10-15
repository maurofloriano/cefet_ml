from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import pandas as pd

class MetricsMixin:

    def get_confusion_matrix(self, results):
        confusion = {}
        for model, result in results.items():
            confusion[model] = pd.DataFrame(
                confusion_matrix(**result),
                columns=['neutro', 'negativo', 'positivo'],
                index=['neutro', 'negativo', 'positivo'],
            )
        return confusion

    def metrics_by_class(self, results):
        by_class = {}
        for model, result in results.items():
            by_class[model] = pd.DataFrame(
                precision_recall_fscore_support(**result),
                index=['Precisão', 'Recall', 'F Score', 'Quantidade'],
                columns=['neutro', 'negativo', 'positivo'],
            )
        return by_class

    def average_metrics(self, results):
        metrics = {}
        for model, result in results.items():
            metrics[model] = (accuracy_score(**result), ) + precision_recall_fscore_support(**result, average='macro')
        return pd.DataFrame(
            metrics,
            index=['Acurácia', 'Precisão', 'Recall', 'F Score', 'Quantidade'],
        )
