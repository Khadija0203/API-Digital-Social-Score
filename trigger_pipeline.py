from google.cloud import aiplatform
import os
import sys

def run_vertex_pipeline():
    PROJECT_ID = "simplifia-hackathon"
    REGION = "us-central1"
    TEMPLATE_PATH = "gs://bucketdeployia/compiled-pipelines/pipeline-definition.json"
    PIPELINE_ROOT = "gs://bucketdeployia/pipeline-root/"
    RAW_DATA_PATH = "gs://bucketdeployia/data/train_toxic_10k.csv"

    print("Initialisation de Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Vérification que le template existe
    if not TEMPLATE_PATH.startswith("gs://"):
        print(" ERREUR : Chemin du template incorrect. Utilise un chemin GCS complet.")
        sys.exit(1)

    print("Fichier pipeline trouvé :", TEMPLATE_PATH)

    try:
        print(" Lancement du job Vertex AI Pipeline...")
        pipeline_job = aiplatform.PipelineJob(
            display_name="toxic-detection-train-pipeline",
            template_path=TEMPLATE_PATH,
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "raw_csv_path": RAW_DATA_PATH
            }
        )

        pipeline_job.submit()
        print(" Pipeline Vertex AI déclenché avec succès.")
        print(f" Suivre le run ici : {pipeline_job.resource_name}")

    except Exception as e:
        print(" ERREUR lors du lancement du pipeline :", e)
        sys.exit(1)


if __name__ == "__main__":
    run_vertex_pipeline()
