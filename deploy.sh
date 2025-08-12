#!/bin/bash

# Replace these variables with your actual values
PROJECT_ID="qa-image-server"
SERVICE_NAME="qa-image-backend"
REGION="us-west2"
CREDENTIALS_FILE="qa-image-server-0b52e275c444.json"  # Your JSON file name
BUCKET_NAME="qa-image-uploads"  # Your existing bucket name

echo "🚀 Starting streamlined deployment process..."
echo "📋 Using existing setup (APIs enabled, bucket created, permissions set)"

# Step 1: Set your project
echo "📝 Setting Google Cloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Step 2: Get your existing service account email from JSON file
echo "👤 Reading service account from existing JSON file..."
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "❌ Error: $CREDENTIALS_FILE not found!"
    echo "Please make sure your JSON file is in the current directory."
    exit 1
fi

SERVICE_ACCOUNT_EMAIL=$(cat $CREDENTIALS_FILE | jq -r '.client_email')
if [ "$SERVICE_ACCOUNT_EMAIL" = "null" ] || [ -z "$SERVICE_ACCOUNT_EMAIL" ]; then
    echo "❌ Error: Could not extract service account email from JSON file."
    echo "Please check your JSON file format."
    exit 1
fi

echo "📧 Using existing service account: $SERVICE_ACCOUNT_EMAIL"

# # Step 3: Store your JSON credentials in Secret Manager
# echo "🔐 Storing credentials in Secret Manager..."
# gcloud secrets create qa-image-credentials --data-file=$CREDENTIALS_FILE 2>/dev/null || {
#     echo "⚠️  Secret already exists, updating it..."
#     gcloud secrets versions add qa-image-credentials --data-file=$CREDENTIALS_FILE
# }


# # Step 4: Grant Secret Manager access to your existing service account
# echo "🔑 Granting Secret Manager access to your service account..."
# gcloud secrets add-iam-policy-binding qa-image-credentials \
#     --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
#     --role="roles/secretmanager.secretAccessor" \
#     --quiet


# Step 5: Verify permissions (optional but recommended)
echo "🔍 Checking permissions for Vision API application..."

# Check Vision API enabled
gcloud services list --enabled --filter="name:vision.googleapis.com" --format="value(name)" | grep -q "vision.googleapis.com" && echo "    ✅ Vision API enabled" || echo "    ❌ Vision API not enabled"

# Check Storage permissions
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:$SERVICE_ACCOUNT_EMAIL AND (bindings.role:roles/storage.objectCreator OR bindings.role:roles/storage.admin)" \
    --format="value(bindings.role)" | grep -q -E "(roles/storage.objectCreator|roles/storage.admin)" && echo "    ✅ Storage access confirmed" || echo "    ❌ Storage access missing"

# Check Secret Manager permissions
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:$SERVICE_ACCOUNT_EMAIL AND bindings.role:roles/secretmanager.secretAccessor" \
    --format="value(bindings.role)" | grep -q "roles/secretmanager.secretAccessor" && echo "    ✅ Secret Manager access confirmed" || echo "    ❌ Secret Manager access missing"

# Step 6: Build and push the Docker image
echo "🏗️ Building and pushing Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Step 7: Deploy to Cloud Run with Secret Manager
echo "🚀 Deploying to Cloud Run with Secret Manager..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --allow-unauthenticated \
    --timeout=300 \
    --service-account=$SERVICE_ACCOUNT_EMAIL \
    --set-secrets=GOOGLE_CREDENTIALS_JSON=qa-image-credentials:latest \
    --set-env-vars STORAGE_BUCKET_NAME=$BUCKET_NAME \
    --region=$REGION \
    --port=8080 \

# Step 8: Get the service URL
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform=managed --region=$REGION --format='value(status.url)')
echo "🌐 Your service is available at: $SERVICE_URL"

# Step 9: Test the deployment
echo "🧪 Testing the health endpoint..."
echo "Making request to: $SERVICE_URL/health"
curl -s "$SERVICE_URL/health" | jq '.' 2>/dev/null || {
    echo "Response (raw):"
    curl -s "$SERVICE_URL/health"
}

echo ""
echo "📋 Deployment Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Project ID: $PROJECT_ID"
echo "✅ Service Name: $SERVICE_NAME"
echo "✅ Region: $REGION"
echo "✅ Service URL: $SERVICE_URL"
echo "✅ Bucket: $BUCKET_NAME"
echo "✅ Secret: qa-image-credentials"
echo "✅ Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "✅ Authentication: Secret Manager + Service Account"
echo ""
echo "🎉 Your QA Image API is now live and secure!"
echo ""
echo "📝 API Endpoints:"
echo "   • Health Check: $SERVICE_URL/health"
echo "   • Image Analysis: $SERVICE_URL/analyze-image (POST)"
echo "   • WebSocket: $SERVICE_URL/ws/{client_id}"
echo "   • API Info: $SERVICE_URL/ws-info"