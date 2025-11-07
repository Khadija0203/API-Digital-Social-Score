from kfp import dsl
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas", "scikit-learn"
    ]
)
def train_model_op(clean_csv: dsl.Input[dsl.Dataset], model: dsl.Output[dsl.Model]):
    """Entraîne un modèle SVM sur des commentaires toxiques"""
    df = pd.read_csv(clean_csv.path)
    X, y = df["comment_text"], df["toxic"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("svm", LinearSVC(class_weight="balanced", max_iter=10000))
    ])
    pipeline.fit(X, y)

    with open(model.path, "wb") as f:
        pickle.dump(pipeline, f)
