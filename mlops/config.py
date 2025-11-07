"""
Configuration centralisée pour le pipeline MLOps
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class MLOpsConfig:
    """Configuration centralisée MLOps"""
    
    # Google Cloud
    project_id: str = os.getenv('PROJECT_ID', '')
    region: str = os.getenv('REGION', 'europe-west1')
    
    # Cloud Storage
    bucket_models: str = f"mlops-models-{os.getenv('PROJECT_ID', 'default')}"
    bucket_data: str = f"mlops-data-{os.getenv('PROJECT_ID', 'default')}"
    
    # MLflow
    mlflow_tracking_uri: str = f"gs://mlops-models-{os.getenv('PROJECT_ID', 'default')}/mlflow"
    experiment_name: str = "toxic-detection-svm"
    
    # Vertex AI
    vertex_pipeline_root: str = f"gs://mlops-models-{os.getenv('PROJECT_ID', 'default')}/pipeline-root"
    
    # Modèle
    model_name: str = "toxic-detection-svm"
    min_accuracy_production: float = 0.85
    
    # Données
    data_sources: list = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = [
                "data/train_toxic_10k.csv",
                "data/dataset_cleaned_and_anonymized_10k.csv"
            ]

# Configuration par défaut
config = MLOpsConfig()