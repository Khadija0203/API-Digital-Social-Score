from kfp import compiler
from pipelines.pipeline import toxic_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=toxic_pipeline,
        package_path="pipeline-definition.json"
    )
    print("✅ Pipeline compilé avec succès : pipeline-definition.json")
