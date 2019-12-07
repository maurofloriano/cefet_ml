import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class MetricsMixin:
    def get_confusion_matrix(self, results):
        confusion = {}
        for model, result in results.items():
            confusion[model] = pd.DataFrame(
                confusion_matrix(**result),
                columns=["neutro", "negativo", "positivo"],
                index=["neutro", "negativo", "positivo"],
            )
        return confusion

    def metrics_by_class(self, results):
        by_class = {}
        for model, result in results.items():
            by_class[model] = pd.DataFrame(
                precision_recall_fscore_support(**result),
                index=["Precisão", "Recall", "F Score", "Quantidade"],
                columns=["neutro", "negativo", "positivo"],
            )
        return by_class

    def get_confusion_matrix_neutral_is_positive(self, results):
        confusion = {}
        for model, result in results.items():
            confusion[model] = pd.DataFrame(
                confusion_matrix(**result),
                columns=["positivo", "negativo"],
                index=["positivo", "negativo"],
            )
        return confusion

    def metrics_by_class_neutral_is_positive(self, results):
        by_class = {}
        for model, result in results.items():
            by_class[model] = pd.DataFrame(
                precision_recall_fscore_support(**result),
                index=["Precisão", "Recall", "F Score", "Quantidade"],
                columns=["positivo", "negativo"],
            )
        return by_class

    def average_metrics(self, results):
        metrics = {}
        for model, result in results.items():
            metrics[model] = (
                accuracy_score(**result),
            ) + precision_recall_fscore_support( average="macro", **result)
        return pd.DataFrame(
            metrics, index=["Acurácia", "Precisão", "Recall", "F Score", "Quantidade"]
        )
