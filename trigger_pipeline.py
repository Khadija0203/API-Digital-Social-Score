from google.cloud import aiplatform

def run_vertex_pipeline():
    # ✅ 1. Initialise Vertex AI avec ton projet et ta région
    aiplatform.init(
        project="simplifia-hackathon",
        location="us-central1"  # ← ta région réelle (puisque ton cluster GKE est ici)
    )

    # ✅ 2. Crée et lance le job du pipeline Vertex AI
    pipeline_job = aiplatform.PipelineJob(
        display_name="toxic-detection-train-pipeline",
        template_path="gs://bucketdeployia/compiled-pipelines/pipeline-definition.json",  # ← bon chemin
        pipeline_root="gs://bucketdeployia/pipeline-root/",
        parameter_values={
            "raw_csv_path": "gs://bucketdeployia/data/train_toxic_10k.csv"  # ← doit correspondre à ton paramètre de pipeline
        }
    )

    pipeline_job.submit()
    print("✅ Pipeline Vertex AI déclenché avec succès.")

if __name__ == "__main__":
    run_vertex_pipeline()
