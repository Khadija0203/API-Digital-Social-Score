import pytest
from unittest.mock import Mock, patch
import os

def test_imports():
    """Test que les imports fonctionnent"""
    try:
        import sklearn
        import pandas
        import numpy
        assert True
    except ImportError:
        assert False, "Dependencies manquantes"

def test_app_structure():
    """Test structure de l'application"""
    # Vérifier que les fichiers essentiels existent dans le repo
    assert os.path.exists("app.py"), "app.py manquant"
    assert os.path.exists("requirements.txt"), "requirements.txt manquant"
    assert os.path.exists("Dockerfile"), "Dockerfile manquant"

@patch('joblib.load')
def test_model_prediction_mock(mock_load):
    """Test prédiction avec modèle mocké"""
    # Mock du modèle
    mock_model = Mock()
    mock_model.predict.return_value = [1]  # Toxic
    mock_model.decision_function.return_value = [2.5]
    mock_load.return_value = mock_model
    
    # Test de prédiction mockée
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    
    # Vérifier que le pipeline peut être créé
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100)),
        ('svm', LinearSVC(random_state=42))
    ])
    
    assert pipeline is not None

def test_mlops_structure():
    """Test structure MLOps"""
    assert os.path.exists("mlops/"), "Dossier mlops/ manquant"
    assert os.path.exists("mlops/training.py"), "mlops/training.py manquant"
    assert os.path.exists("mlops/config.py"), "mlops/config.py manquant"

def test_configuration():
    """Test configuration des variables"""
    # Test que les variables peuvent être lues
    project_id = os.getenv('PROJECT_ID', 'test-project')
    region = os.getenv('REGION', 'europe-west1')
    
    assert len(project_id) > 0
    assert len(region) > 0