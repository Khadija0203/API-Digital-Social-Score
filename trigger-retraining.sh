#!/bin/bash

# Script pour déclencher le pipeline de réentraînement automatique
# Usage: ./trigger-retraining.sh

set -e

PROJECT_ID=${PROJECT_ID:-"simplifia-hackathon"}
REGION=${REGION:-"europe-west1"}

echo "════════════════════════════════════════════════════════"
echo "🚀 DÉCLENCHEMENT DU PIPELINE DE RÉENTRAÎNEMENT"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📌 Projet: $PROJECT_ID"
echo "📍 Région: $REGION"
echo ""

# Vérifier que les données sont disponibles
echo "🔍 Vérification des données dans GCS..."
if gsutil ls "gs://mlops-models-$PROJECT_ID/data/" > /dev/null 2>&1; then
    echo "✅ Données trouvées dans GCS"
else
    echo "⚠️  Aucune donnée trouvée dans gs://mlops-models-$PROJECT_ID/data/"
    echo "   Les données seront téléchargées depuis Hugging Face"
fi

echo ""
echo "🔧 Lancement du pipeline Cloud Build..."
gcloud builds submit \
    --config=cloudbuild-retraining.yaml \
    --project=$PROJECT_ID \
    --region=$REGION \
    .

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Pipeline de réentraînement démarré !"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📊 Suivre l'exécution:"
echo "   https://console.cloud.google.com/cloud-build/builds?project=$PROJECT_ID"
echo ""
echo "📈 Vertex AI Pipelines:"
echo "   https://console.cloud.google.com/vertex-ai/pipelines/runs?project=$PROJECT_ID"
echo ""
echo "🔍 MLflow Tracking:"
echo "   gs://mlops-models-$PROJECT_ID/mlflow"
echo ""
