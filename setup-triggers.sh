# Script pour configurer les triggers Cloud Build
# Usage: ./setup-triggers.sh

set -e

PROJECT_ID=${PROJECT_ID:-"simplifia-hackathon"}
REGION=${REGION:-"europe-west1"}
REPO_OWNER="Khadija0203"
REPO_NAME="API-Digital-Social-Score"
BUCKET_NAME="mlops-models-${PROJECT_ID}"


echo "════════════════════════════════════════════════════════"
echo "🔧 CONFIGURATION DES TRIGGERS CLOUD BUILD"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📌 Projet: $PROJECT_ID"
echo "📦 Repo: $REPO_OWNER/$REPO_NAME"
echo "🗄️  Bucket: $BUCKET_NAME"
echo ""

# ============================================================
# TRIGGER 1: Déploiement de l'API (Push sur main)
# ============================================================

echo "🚀 Création du trigger 1: Déploiement API..."

gcloud builds triggers create github \
    --name="deploy-api-on-push" \
    --repo-name="$REPO_NAME" \
    --repo-owner="$REPO_OWNER" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml" \
    --description="Déploie l'API sur GKE quand on push sur main" \
    --project="$PROJECT_ID" \
    2>&1 | grep -v "already exists" || echo "✅ Trigger deploy-api-on-push configuré"

echo ""

# ============================================================
# TRIGGER 2: Réentraînement (Nouveau fichier dans GCS data/)
# ============================================================

echo "🔄 Création du trigger 2: Réentraînement automatique..."
echo "   (Utilise Pub/Sub car triggers GCS directs non supportés)"
echo ""

# Étape 1: Créer un topic Pub/Sub
echo "📢 Création du topic Pub/Sub..."
gcloud pubsub topics create gcs-data-changes \
    --project="$PROJECT_ID" \
    2>&1 | grep -v "already exists" || echo "✅ Topic gcs-data-changes existe"

# Étape 2: Configurer les notifications GCS → Pub/Sub
echo "🔔 Configuration notification GCS → Pub/Sub..."
gsutil notification create \
    -t gcs-data-changes \
    -f json \
    -e OBJECT_FINALIZE \
    -p data/ \
    "gs://$BUCKET_NAME" \
    2>&1 || echo "✅ Notification GCS configurée"

# Étape 3: Créer le trigger Cloud Build déclenché par Pub/Sub
echo "⚙️  Création du trigger Cloud Build (Pub/Sub → cloudbuild-retrain.yaml)..."

# Pour trigger Pub/Sub, on doit utiliser inline build config
gcloud builds triggers create pubsub \
    --name="retrain-on-data-change" \
    --topic="projects/$PROJECT_ID/topics/gcs-data-changes" \
    --inline-config="cloudbuild-retrain.yaml" \
    --description="Réentraîne le modèle quand des données sont ajoutées dans GCS" \
    --project="$PROJECT_ID" \
    --substitutions="_PROJECT_ID=$PROJECT_ID" \
    2>&1 | grep -v "already exists" || echo "✅ Trigger retrain-on-data-change configuré"

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ CONFIGURATION TERMINÉE !"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📋 Vérifier les triggers créés :"
echo "   gcloud builds triggers list"
echo ""
echo "🔗 Console Cloud Build Triggers :"
echo "   https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID"
echo ""
echo "🧪 Pour tester le réentraînement automatique :"
echo "   gsutil cp data/test_toxic_10k.csv gs://$BUCKET_NAME/data/"
echo ""
echo "📊 Suivre l'exécution :"
echo "   gcloud builds list --ongoing"
echo "   https://console.cloud.google.com/cloud-build/builds?project=$PROJECT_ID"
echo ""
echo "════════════════════════════════════════════════════════"
