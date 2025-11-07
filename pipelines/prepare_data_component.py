from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "gcsfs"]
)
def prepare_data_op(raw_csv_path: str, clean_csv: dsl.Output[dsl.Dataset]):
    """Nettoyage du jeu de données brut"""
    import pandas as pd
    df = pd.read_csv(raw_csv_path)
    df.dropna(subset=["comment_text"], inplace=True)
    df.to_csv(clean_csv.path, index=False)
