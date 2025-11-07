from google.cloud import aiplatform

def run_vertex_pipeline():
    aiplatform.init(project="simplifia-hackathon", location="europe-west1")

    pipeline_job = aiplatform.PipelineJob(
        display_name="toxic-detection-train-pipeline",
        template_path="gs://bucketdeployia/pipeline-definition.json",
        pipeline_root="gs://bucketdeployia/pipeline-root/",
        parameter_values={
            "data_path": "gs://bucketdeployia/data/train_toxic_10k.csv"
        }
    )

    pipeline_job.submit()
    print("✅ Pipeline Vertex AI déclenché avec succès.")

if __name__ == "__main__":
    run_vertex_pipeline()
