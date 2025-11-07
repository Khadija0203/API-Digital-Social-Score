from kfp import dsl
import pandas as pd

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas"]
)
def prepare_data_op(raw_csv_path: str, clean_csv: dsl.Output[dsl.Dataset]):
    """Nettoyage du jeu de données brut"""
    df = pd.read_csv(raw_csv_path)
    df.dropna(subset=["comment_text"], inplace=True)
    df.to_csv(clean_csv.path, index=False)
