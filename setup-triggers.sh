# Script pour configurer les triggers Cloud Build
# Usage: ./setup-triggers.sh

set -e

PROJECT_ID=${PROJECT_ID:-"simplifia-hackathon"}
REGION=${REGION:-"europe-west1"}
REPO_OWNER="Khadija0203"
REPO_NAME="API-Digital-Social-Score"
BUCKET_NAME="mlops-models-${PROJECT_ID}"


echo ""
echo "Projet: $PROJECT_ID"
echo "Repo: $REPO_OWNER/$REPO_NAME"
echo "Bucket: $BUCKET_NAME"
echo ""

# ============================================================
# TRIGGER 1: Déploiement de l'API (Push sur main)
# ============================================================

echo " Création du trigger 1: Déploiement API..."

gcloud builds triggers create github \
    --name="deploy-api-on-push" \
    --repo-name="$REPO_NAME" \
    --repo-owner="$REPO_OWNER" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml" \
    --description="Déploie l'API sur GKE quand on push sur main" \
    --region="$REGION" \
    --substitutions="_PROJECT_ID=$PROJECT_ID" \
    || echo "  Trigger deploy-api-on-push existe déjà"

echo " Trigger 1 configuré !"
echo ""

# ============================================================
# TRIGGER 2: Réentraînement (Nouveau fichier dans GCS data/)
# ============================================================

echo " Création du trigger 2: Réentraînement automatique..."

# Note: Cloud Build Triggers ne supportent pas directement GCS events
# Il faut utiliser Pub/Sub + Cloud Storage notifications

echo "  Les triggers GCS directs ne sont pas supportés par Cloud Build"
echo " Utilisation d'une approche alternative avec Pub/Sub..."

# Créer un topic Pub/Sub
gcloud pubsub topics create gcs-data-changes \
    --project="$PROJECT_ID" \
    || echo "Topic gcs-data-changes existe déjà"

# Configurer les notifications Cloud Storage vers Pub/Sub
gsutil notification create \
    -t gcs-data-changes \
    -f json \
    -e OBJECT_FINALIZE \
    -p data/ \
    "gs://$BUCKET_NAME" \
    || echo "Notification déjà configurée"

# Créer le trigger Cloud Build déclenché par Pub/Sub
gcloud builds triggers create pubsub \
    --name="retrain-on-data-change" \
    --topic="gcs-data-changes" \
    --repo-name="$REPO_NAME" \
    --repo-owner="$REPO_OWNER" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild-retrain.yaml" \
    --description="Réentraîne le modèle quand des données sont ajoutées dans GCS" \
    --region="$REGION" \
    --substitutions="_PROJECT_ID=$PROJECT_ID" \
    || echo "  Trigger retrain-on-data-change existe déjà"

echo " Trigger 2 configuré !"
echo ""
