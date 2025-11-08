from kfp import dsl
from pipelines.prepare_data_component import prepare_data_op
from pipelines.train_model_component import train_model_op
from pipelines.evaluate_model_component import evaluate_model_op

@dsl.pipeline(
    name="toxic-detection-pipeline",
    pipeline_root="gs://bucketdeployia/pipeline-root"
)
def toxic_pipeline(raw_csv_path: str = "gs://bucketdeployia/data/train_toxic_10k.csv"):
    """Pipeline complet : Préparation -> Entraînement -> Évaluation"""

    # 1. Préparation des données
    prep = prepare_data_op(raw_csv_path=raw_csv_path)

    # 2. Entraînement du modèle
    train = train_model_op(clean_csv=prep.outputs["clean_csv"])
    train.after(prep)

    # 3. Évaluation du modèle
    eval = evaluate_model_op(model=train.outputs["model"], clean_csv=prep.outputs["clean_csv"])
    eval.after(train)
    print("Pipeline de détection de toxicité prêt à l’exécution !")
