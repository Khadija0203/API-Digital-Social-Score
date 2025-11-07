from kfp.v2.dsl import component, Input, Model, Dataset
import pandas as pd, pickle
from sklearn.metrics import accuracy_score

@component(base_image="python:3.11")
def evaluate_model_op(model: Input[Model], clean_csv: Input[Dataset]) -> float:
    df = pd.read_csv(clean_csv.path)
    with open(model.path, "rb") as f:
        pipe = pickle.load(f)
    preds = pipe.predict(df["comment_text"])
    acc = accuracy_score(df["toxic"], preds)
    print(f"✅ Accuracy: {acc:.4f}")
    return acc
