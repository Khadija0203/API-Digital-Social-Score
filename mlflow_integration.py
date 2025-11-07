"""
Modification de app.py pour integration MLflow
Charge automatiquement le modele depuis MLflow Registry
"""

import mlflow
import mlflow.sklearn
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MLflowModelManager:
    """Gestionnaire de modeles via MLflow Registry"""
    
    def __init__(self, 
                 model_name: str = "toxic-detection-svm",
                 stage: str = "Production",
                 fallback_path: str = "./model/svm_model.pkl"):
        self.model_name = model_name
        self.stage = stage
        self.fallback_path = fallback_path
        self.model = None
        self.model_version = None
        self.model_uri = None
        
        # Configuration MLflow
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'gs://mlops-models-{}/mlflow'.format(os.getenv('PROJECT_ID')))
        mlflow.set_tracking_uri(mlflow_uri)
    
    def load_model(self) -> bool:
        """
        Charge le modele depuis MLflow Registry avec fallback
        
        Returns:
            bool: True si succes, False sinon
        """
        try:
            # Tentative de chargement depuis MLflow Registry
            model_uri = f"models:/{self.model_name}/{self.stage}"
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_uri = model_uri
            
            # Obtenir les metadonnees de la version
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            latest_version = client.get_latest_versions(self.model_name, stages=[self.stage])
            if latest_version:
                self.model_version = latest_version[0].version
                run_id = latest_version[0].run_id
                
                # Log des informations du modele
                logger.info(f"Modele MLflow charge:")
                logger.info(f"  - URI: {model_uri}")
                logger.info(f"  - Version: {self.model_version}")
                logger.info(f"  - Run ID: {run_id}")
                
                # Obtenir les metriques du modele
                run = client.get_run(run_id)
                test_accuracy = run.data.metrics.get('test_accuracy')
                if test_accuracy:
                    logger.info(f"  - Test Accuracy: {test_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Impossible de charger depuis MLflow: {e}")
            
            # Fallback vers modele local
            try:
                import pickle
                with open(self.fallback_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                self.model_uri = f"local://{self.fallback_path}"
                logger.info(f"Modele local charge en fallback: {self.fallback_path}")
                return True
                
            except Exception as e2:
                logger.error(f"Impossible de charger le modele local: {e2}")
                return False
    
    def predict(self, texts):
        """Prediction avec le modele charge"""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        
        return self.model.predict(texts)
    
    def predict_proba(self, texts):
        """Prediction avec probabilites"""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        
        try:
            return self.model.predict_proba(texts)
        except AttributeError:
            # Si predict_proba non disponible, utiliser decision_function
            from scipy.special import expit
            decision_scores = self.model.decision_function(texts)
            return [[1-expit(score), expit(score)] for score in decision_scores]
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modele"""
        return {
            'model_name': self.model_name,
            'stage': self.stage,
            'version': self.model_version,
            'uri': self.model_uri,
            'loaded': self.model is not None
        }
    
    def refresh_model(self) -> bool:
        """Recharge le modele (pour mise a jour)"""
        logger.info("Rafraichissement du modele...")
        return self.load_model()

# Modification a apporter dans app.py
"""
Remplacer la section de chargement du modele dans app.py par:

# Gestionnaire de modele MLflow
model_manager = MLflowModelManager()
model_loaded = model_manager.load_model()

def load_model():
    global model_manager, model_loaded
    try:
        model_loaded = model_manager.load_model()
        if model_loaded:
            logger.info("Modele MLflow charge avec succes")
            MODEL_STATUS.set(1)
        else:
            logger.error("Echec chargement modele MLflow")
            MODEL_STATUS.set(0)
            PREDICTION_ERRORS.labels(error_type="model_load").inc()
        return model_loaded
    except Exception as e:
        logger.error(f"Erreur chargement modele: {str(e)}")
        MODEL_STATUS.set(0)
        PREDICTION_ERRORS.labels(error_type="model_load").inc()
        return False

# Dans la fonction predict_toxicity, remplacer:
prediction = int(model.predict([text])[0])
# par:
prediction = int(model_manager.predict([text])[0])

# Et pour les probabilites:
probabilities = model_manager.predict_proba([text])[0]
probability_toxic = float(probabilities[1])

# Ajouter un endpoint pour info modele:
@app.get("/model/info")
async def model_info():
    return model_manager.get_model_info()

# Ajouter un endpoint pour recharger le modele:
@app.post("/model/refresh")
async def refresh_model():
    success = model_manager.refresh_model()
    return {"success": success, "model_info": model_manager.get_model_info()}
"""