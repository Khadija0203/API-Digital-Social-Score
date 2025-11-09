# Integre tracking, registry et gestion des versions
import os
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import pickle
from datetime import datetime
import json
from google.cloud import storage
import io
from mlflow.tracking import MlflowClient
from google.cloud import storage
import tempfile

# Configuration MLflow - compatible Cloud Build
PROJECT_ID = os.getenv('PROJECT_ID')
EXPERIMENT_NAME = "toxic-detection-svm"
LOCAL_MLFLOW_DIR = "/tmp/mlflow"
GCS_MLFLOW_BUCKET = f"mlops-models-{PROJECT_ID}"

def setup_mlflow():
    """Configure MLflow avec tracking local + sync GCS"""
    
    # Utiliser tracking local (compatible avec tous environnements)
    local_tracking_uri = f"file://{LOCAL_MLFLOW_DIR}"
    mlflow.set_tracking_uri(local_tracking_uri)
    
    print(f"🔧 MLflow configuré avec URI local: {local_tracking_uri}")
    
    # Créer ou obtenir l'expérience
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=f"{LOCAL_MLFLOW_DIR}/artifacts/{EXPERIMENT_NAME}"
            )
            print(f"✅ Expérience créée: {EXPERIMENT_NAME} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Expérience existante: {EXPERIMENT_NAME} (ID: {experiment_id})")
            
        mlflow.set_experiment(experiment_id=experiment_id)
        return experiment_id
        
    except Exception as e:
        print(f"❌ Erreur configuration MLflow: {e}")
        # Fallback vers expérience par défaut
        return "0"

def sync_mlflow_to_gcs():
    """Synchroniser les artefacts MLflow locaux vers GCS"""
    try:
        print(f"📦 Synchronisation MLflow vers gs://{GCS_MLFLOW_BUCKET}/mlflow...")
        
        # Utiliser gsutil pour synchroniser (plus fiable que l'API Python)
        import subprocess
        cmd = [
            "gsutil", "-m", "rsync", "-r", "-d", 
            LOCAL_MLFLOW_DIR, 
            f"gs://{GCS_MLFLOW_BUCKET}/mlflow"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Synchronisation MLflow réussie vers GCS")
            return True
        else:
            print(f"⚠️ Erreur sync MLflow: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"⚠️ Erreur sync MLflow vers GCS: {e}")
        return False

def train_svm_with_mlflow(bucket_name=None, data_blob_path="data/train_toxic_10k.csv"):
    """
    Entrainement SVM avec tracking MLflow complet
    Utilise Cloud Storage pour les donnees
    """
    
    # Configuration MLflow
    experiment_id = setup_mlflow()
    
    # Configuration Cloud Storage
    if bucket_name is None:
        project_id = os.getenv('PROJECT_ID')
        bucket_name = f"mlops-models-{project_id}"
    
    with mlflow.start_run(run_name=f"svm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        
        print("Chargement des données depuis Cloud Storage...")
        
        # Chargement des donnees depuis Cloud Storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Essayer plusieurs sources de donnees
        data_sources = [
            "data/train_toxic_10k.csv",
            "data/dataset_cleaned_and_anonymized_10k.csv", 
        ]
        
        df = None
        used_source = None
        
        for source in data_sources:
            try:
                blob = bucket.blob(source)
                content = blob.download_as_text()
                df = pd.read_csv(io.StringIO(content))
                used_source = source
                print(f"Données chargées depuis gs://{bucket_name}/{source}")
                break
            except Exception as e:
                print(f"Impossible de charger {source}: {e}")
                continue
        
        if df is None:
            raise ValueError("Aucune source de données trouvée dans Cloud Storage")
        
        # Verification des colonnes
        if 'comment_text' not in df.columns or 'toxic' not in df.columns:
            print("Ajustement des noms de colonnes...")
            # Mapping automatique des colonnes
            text_col = None
            label_col = None
            
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['text', 'comment', 'message']):
                    text_col = col
                if any(keyword in col.lower() for keyword in ['toxic', 'label', 'target']):
                    label_col = col
            
            if text_col and label_col:
                df = df.rename(columns={text_col: 'comment_text', label_col: 'toxic'})
                print(f"Colonnes renommées: {text_col} -> comment_text, {label_col} -> toxic")
        
        texts = df['comment_text'].astype(str).tolist()
        labels = df['toxic'].tolist()
        
        print(f"Données chargées: {len(texts)} commentaires")
        print(f"Distribution: {df['toxic'].value_counts().to_dict()}")
        
        # Log des donnees dans MLflow
        mlflow.log_param("dataset_size", len(texts))
        mlflow.log_param("data_source", f"gs://{bucket_name}/{used_source}")
        mlflow.log_param("bucket_name", bucket_name)
        mlflow.log_param("toxic_ratio", df['toxic'].mean())
        
        # Division train/test pour evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # PIPELINE TF-IDF + SVM (logique identique SVM.py)
        print("Configuration du pipeline TF-IDF + SVM...")
        
        # Parametres du modele
        tfidf_params = {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'strip_accents': 'ascii',
            'lowercase': True,
            'stop_words': 'english'
        }
        
        svm_params = {
            'random_state': 42,
            'class_weight': 'balanced',
            'max_iter': 10000
        }
        
        # Log des hyperparametres
        for param, value in tfidf_params.items():
            mlflow.log_param(f"tfidf_{param}", value)
        
        for param, value in svm_params.items():
            mlflow.log_param(f"svm_{param}", value)
        
        # Creation du pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('svm', LinearSVC(**svm_params))
        ])
        
        # ENTRAINEMENT
        print("Début de l'entraînement SVM...")
        start_time = datetime.now()
        
        pipeline.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("training_time_seconds", training_time)
        
        print("Entraînement terminé!")
        
        # EVALUATION
        print("Evaluation du modele : ")
        
        # Predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Metriques d'entrainement
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_precision = precision_score(y_train, y_pred_train, average='weighted')
        train_recall = recall_score(y_train, y_pred_train, average='weighted')
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        
        # Metriques de test
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Log des metriques dans MLflow
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)
        
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        mlflow.log_metric("cv_accuracy_mean", cv_mean)
        mlflow.log_metric("cv_accuracy_std", cv_std)
        
        # Rapport de classification
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Log du rapport comme artefact
        with open("classification_report.json", "w") as f:
            json.dump(class_report, f, indent=2)
        mlflow.log_artifact("classification_report.json")
        
        # SAUVEGARDE DU MODELE
        print("Sauvegarde du modèle...")
        
        # Sauvegarde dans Cloud Storage
        run_id = run.info.run_id
        model_filename = f"svm_model_{run_id}.pkl"
        
        # Sauvegarde temporaire locale
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            pickle.dump(pipeline, temp_file)
            temp_model_path = temp_file.name
        
        # Upload vers Cloud Storage
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(temp_model_path)
        
        model_gcs_path = f"gs://{bucket_name}/models/{model_filename}"
        
        # Nettoyage du fichier temporaire
        os.unlink(temp_model_path)
        
        # Log du chemin Cloud Storage
        mlflow.log_param("model_gcs_path", model_gcs_path)
        
        # Enregistrement dans MLflow Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="toxic-detection-svm",
            input_example=X_test[:5],
            signature=mlflow.models.infer_signature(X_train, y_pred_train)
        )
        
        print(f"Modèle sauvegardé dans Cloud Storage: {model_gcs_path}")
        
        # Tags pour organiser les runs
        mlflow.set_tag("model_type", "svm")
        mlflow.set_tag("algorithm", "LinearSVC")
        mlflow.set_tag("feature_extraction", "TfIdf")
        mlflow.set_tag("data_version", "10k_anonymized")
        mlflow.set_tag("environment", "vertex_ai")
        
        print("ENTRAÎNEMENT SVM TERMINÉ!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Retourner les metriques pour validation
        return {
            'run_id': run.info.run_id,
            'model_uri': model_info.model_uri,
            'test_accuracy': test_accuracy,
            'cv_accuracy': cv_mean,
            'model_version': model_info.model_uri.split('/')[-1] if model_info.model_uri else None
        }

def promote_model_to_production(model_name="toxic-detection-svm", min_accuracy=0.85):
    """
    Promotion automatique du modele vers Production si metriques OK
    """
    
    client = MlflowClient()
    
    # Obtenir la derniere version du modele en Staging
    latest_versions = client.get_latest_versions(
        model_name, 
        stages=["Staging", "None"]
    )
    
    if not latest_versions:
        print("Aucune version du modele trouvee")
        return False
    
    latest_version = latest_versions[0]
    
    # Obtenir les metriques du run
    run = client.get_run(latest_version.run_id)
    test_accuracy = run.data.metrics.get('test_accuracy', 0)
    
    print(f"Version {latest_version.version}: Test Accuracy = {test_accuracy:.4f}")
    
    if test_accuracy >= min_accuracy:
        # Promouvoir vers Production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"Modele version {latest_version.version} promu vers Production!")
        return True
    else:
        print(f"Accuracy {test_accuracy:.4f} < {min_accuracy:.4f}, promotion refusee")
        return False

def load_model_from_gcs(bucket_name, model_path):
    """
    Charge un modele depuis Cloud Storage
    """
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Telecharger le modele dans un fichier temporaire
    blob = bucket.blob(model_path)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        
        # Charger le modele
        with open(temp_file.name, 'rb') as f:
            model = pickle.load(f)
        
        # Nettoyage
        os.unlink(temp_file.name)
        
        return model

def load_production_model(bucket_name=None, model_name="toxic-detection-svm"):
    """
    Charge le modele de Production depuis MLflow Registry et Cloud Storage
    """
    if bucket_name is None:
        project_id = os.getenv('PROJECT_ID')
        bucket_name = f"mlops-models-{project_id}"
    
    # Methode 1: MLflow Registry (prioritaire)
    model_uri = f"models:/{model_name}/Production"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Modele Production charge depuis MLflow: {model_uri}")
        return model
    except Exception as e:
        print(f"Erreur chargement MLflow: {e}")
    
    # Methode 2: Cloud Storage (fallback)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            run_id = prod_versions[0].run_id
            model_path = f"models/svm_model_{run_id}.pkl"
            
            model = load_model_from_gcs(bucket_name, model_path)
            print(f"Modele Production charge depuis Cloud Storage: gs://{bucket_name}/{model_path}")
            return model
    except Exception as e:
        print(f"Erreur chargement Cloud Storage: {e}")
    
    print("Aucune méthode de chargement disponible")
    return None

if __name__ == "__main__":
    import sys
    
    # Mode rapide pour développement
    quick_mode = "--quick-mode" in sys.argv
    
    try:
        if quick_mode:
            print("MODE RAPIDE - Echantillon reduit pour developpement")
            # Utiliser moins de données pour test rapide
            bucket_name = f"mlops-data-{os.getenv('PROJECT_ID')}"
            result = train_svm_with_mlflow(bucket_name=bucket_name)
        else:
            print("MODE COMPLET - Entrainement MLflow standard")
            result = train_svm_with_mlflow()
        
        # Synchroniser MLflow vers GCS après l'entraînement
        if result:
            print("🔄 Synchronisation des artefacts MLflow vers GCS...")
            sync_success = sync_mlflow_to_gcs()
            
            if sync_success:
                print("✅ Artefacts MLflow sauvegardés dans GCS")
                
                # Promotion automatique si accuracy > 85%
                if result.get('test_accuracy', 0) > 0.85:
                    print(f"🚀 Accuracy {result['test_accuracy']:.4f} > 0.85, promotion du modèle...")
                    promote_model_to_production()
                else:
                    print(f"📊 Accuracy {result.get('test_accuracy', 0):.4f} < 0.85, pas de promotion")
            else:
                print("⚠️ Erreur synchronisation GCS, modèle sauvé localement uniquement")
        else:
            print("❌ Échec de l'entraînement, aucune sauvegarde")
            
    except Exception as e:
        print(f"❌ Erreur dans le pipeline d'entraînement: {e}")
        import traceback
        traceback.print_exc()
    else:
        print("Accuracy insuffisante pour promotion automatique")