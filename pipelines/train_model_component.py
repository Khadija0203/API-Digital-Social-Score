from kfp import dsl
@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas", "scikit-learn", "gcsfs"  # 🔥 ajoute gcsfs si le Dataset est sur GCS
    ]
)
def train_model_op(clean_csv: dsl.Input[dsl.Dataset], model: dsl.Output[dsl.Model]):
    import pandas as pd, pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    # Chargement des données
    df = pd.read_csv(clean_csv.path)
    if "comment_text" not in df.columns or "toxic" not in df.columns:
        raise ValueError(" Les colonnes 'comment_text' et 'toxic' sont requises.")

    X, y = df["comment_text"], df["toxic"]

    # Entraînement
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("svm", LinearSVC(class_weight="balanced", max_iter=10000))
    ])
    pipeline.fit(X, y)

    # Sauvegarde du modèle
    with open(model.path, "wb") as f:
        pickle.dump(pipeline, f)
    print(" Modèle SVM entraîné et sauvegardé :", model.path)
