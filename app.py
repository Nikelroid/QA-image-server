# app.py - Updated for modern FastAPI with lifespan events
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import json
import base64
import uuid
import asyncio
import os
import logging
import sys
from google.cloud import vision
from google.cloud import storage
from google.cloud import secretmanager
from google.oauth2 import service_account
import io
from PIL import Image

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables for clients
vision_client = None
storage_client = None
credentials = None
bucket_name = None

# Initialize Google Cloud clients with Secret Manager support
def get_google_credentials():
    """Get Google credentials from Secret Manager, environment variable, or default"""
    
    # Try Secret Manager first (for Cloud Run)
    try:
        secret_name = os.getenv('GOOGLE_CREDENTIALS_SECRET', 'qa-image-credentials')
        if secret_name:
            # Get project ID from metadata server or environment
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT')
            
            if project_id:
                client = secretmanager.SecretManagerServiceClient()
                secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
                
                response = client.access_secret_version(request={"name": secret_path})
                credentials_json = response.payload.data.decode('utf-8')
                
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                logger.info("✅ Using credentials from Secret Manager")
                return credentials
    except Exception as e:
        logger.warning(f"Failed to get credentials from Secret Manager: {e}")
    
    # Try environment variable (for local development)
    credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
    if credentials_json:
        try:
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            logger.info("✅ Using credentials from GOOGLE_CREDENTIALS_JSON environment variable")
            return credentials
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in GOOGLE_CREDENTIALS_JSON: {e}")
    
    # Default credentials (service account attached to Cloud Run)
    logger.info("✅ Using default Google Cloud credentials")
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan event handler"""
    # Startup
    logger.info("🚀 QA Image API starting up...")
    
    try:
        global vision_client, storage_client, credentials, bucket_name
        
        # Get credentials
        credentials = get_google_credentials()
        
        # Initialize Google Cloud clients
        if credentials:
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            storage_client = storage.Client(credentials=credentials)
        else:
            # This will use the service account automatically on Cloud Run
            vision_client = vision.ImageAnnotatorClient()
            storage_client = storage.Client()
        
        logger.info("✅ Google Cloud clients initialized successfully")
        
        # Get bucket name from environment or use default
        bucket_name = os.getenv('STORAGE_BUCKET_NAME', 'qa-image-uploads')
        logger.info(f"📦 Using bucket: {bucket_name}")
        logger.info(f"🔐 Credentials source: {'secret_manager' if credentials else 'default_service_account'}")
        logger.info("✅ Startup complete")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        # Don't raise the exception - let the app start but log the error
        # This prevents startup failures due to credential issues
    
    yield
    
    # Shutdown
    logger.info("🛑 QA Image API shutting down...")
    # Add any cleanup code here if needed

# Create FastAPI app with lifespan
app = FastAPI(
    title="QA Image API", 
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"🔌 Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"🔌 Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"❌ Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

# Pydantic models
class ImageAnalysisRequest(BaseModel):
    image: str  # base64 encoded image
    question: str

class ImageAnalysisResponse(BaseModel):
    answer: str
    image_url: Optional[str] = None
    vision_data: Dict[str, Any]

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]

# Health check endpoint (required for Cloud Run)
@app.get("/health")
async def health_check():
    """Health check endpoint with service information"""
    try:
        # Test basic functionality
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'unknown')
        
        return {
            "status": "healthy",
            "service": "qa-image-backend",
            "version": "1.0.0",
            "project_id": project_id,
            "bucket": bucket_name or "not_configured",
            "credentials_source": "secret_manager" if credentials else "default_service_account",
            "active_connections": len(manager.active_connections),
            "environment": "production" if os.getenv('GAE_ENV') else "development",
            "clients_initialized": vision_client is not None and storage_client is not None
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "QA Image Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze_image": "/analyze-image (POST)",
            "websocket": "/ws/{client_id}",
            "websocket_info": "/ws-info"
        },
        "features": [
            "Image analysis with Google Vision API",
            "Real-time WebSocket communication",
            "Cloud Storage integration",
            "Multi-format image support"
        ]
    }

# REST API endpoint
@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze image and answer questions about it"""
    try:
        if not vision_client or not storage_client:
            raise HTTPException(status_code=503, detail="Google Cloud services not initialized")
            
        logger.info(f"🔍 Analyzing image with question: {request.question}")
        
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        logger.info(f"📷 Image decoded, size: {len(image_data)} bytes")
        
        # Upload image to Cloud Storage
        image_url = await upload_image_to_storage(image_data)
        
        # Analyze image with Vision API
        vision_results = await analyze_with_vision_api(image_data)
        
        # Generate answer
        answer = await generate_answer(vision_results, request.question)
        
        logger.info(f"✅ Analysis complete")
        
        return ImageAnalysisResponse(
            answer=answer,
            image_url=image_url,
            vision_data=vision_results
        )
        
    except Exception as e:
        logger.error(f"❌ Error in analyze_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time image analysis"""
    await manager.connect(websocket, client_id)
    
    try:
        # Send connection confirmation
        await manager.send_personal_message({
            "type": "connection_confirmed",
            "data": {
                "message": "Connected successfully", 
                "client_id": client_id,
                "server_info": {
                    "version": "1.0.0",
                    "features": ["image_analysis", "real_time_processing"]
                }
            }
        }, client_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(message, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"❌ WebSocket error for client {client_id}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "data": {"message": str(e)}
        }, client_id)
        manager.disconnect(client_id)

async def handle_websocket_message(message: dict, client_id: str):
    """Handle different types of WebSocket messages"""
    message_type = message.get("type")
    data = message.get("data", {})
    
    if message_type == "analyze_image":
        await handle_image_analysis(data, client_id)
    elif message_type == "ping":
        await manager.send_personal_message({
            "type": "pong",
            "data": {"timestamp": data.get("timestamp")}
        }, client_id)
    else:
        await manager.send_personal_message({
            "type": "error",
            "data": {"message": f"Unknown message type: {message_type}"}
        }, client_id)

async def handle_image_analysis(data: dict, client_id: str):
    """Handle image analysis request via WebSocket"""
    try:
        if not vision_client or not storage_client:
            await manager.send_personal_message({
                "type": "error",
                "data": {"message": "Google Cloud services not initialized"}
            }, client_id)
            return
            
        # Send processing status
        await manager.send_personal_message({
            "type": "status",
            "data": {"message": "Processing image...", "status": "processing"}
        }, client_id)
        
        image_base64 = data.get("image")
        question = data.get("question", "What do you see in this image?")
        
        if not image_base64:
            raise ValueError("No image data provided")
        
        logger.info(f"🔍 WebSocket image analysis for client {client_id}")
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Upload image to Cloud Storage (optional for WebSocket)
        upload_task = asyncio.create_task(upload_image_to_storage(image_data))
        
        # Analyze image with Vision API
        vision_results = await analyze_with_vision_api(image_data)
        
        # Send intermediate results
        await manager.send_personal_message({
            "type": "vision_analysis",
            "data": {"vision_data": vision_results}
        }, client_id)
        
        # Generate answer
        answer = await generate_answer(vision_results, question)
        
        # Wait for upload to complete
        image_url = await upload_task
        
        # Send final result
        await manager.send_personal_message({
            "type": "analysis_complete",
            "data": {
                "answer": answer,
                "image_url": image_url,
                "vision_data": vision_results,
                "question": question
            }
        }, client_id)
        
    except Exception as e:
        logger.error(f"❌ WebSocket analysis error: {e}")
        await manager.send_personal_message({
            "type": "error",
            "data": {"message": str(e)}
        }, client_id)

async def upload_image_to_storage(image_data: bytes) -> Optional[str]:
    """Upload image to Google Cloud Storage"""
    try:
        if not storage_client or not bucket_name:
            logger.warning("Storage client or bucket not configured")
            return None
            
        def upload_sync():
            bucket = storage_client.bucket(bucket_name)
            blob_name = f"images/{uuid.uuid4()}.jpg"
            blob = bucket.blob(blob_name)
            
            blob.upload_from_string(image_data, content_type='image/jpeg')
            blob.make_public()
            
            return blob.public_url
        
        # Run the synchronous upload in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, upload_sync)
        logger.info(f"📤 Image uploaded successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error uploading image: {e}")
        return None

async def analyze_with_vision_api(image_data: bytes) -> dict:
    """Analyze image using Google Cloud Vision API"""
    try:
        if not vision_client:
            return {"error": "Vision client not initialized"}
            
        def analyze_sync():
            image = vision.Image(content=image_data)
            
            # Perform different types of analysis
            response = vision_client.annotate_image({
                'image': image,
                'features': [
                    {'type_': vision.Feature.Type.LABEL_DETECTION, 'max_results': 10},
                    {'type_': vision.Feature.Type.TEXT_DETECTION},
                    {'type_': vision.Feature.Type.OBJECT_LOCALIZATION, 'max_results': 10},
                    {'type_': vision.Feature.Type.FACE_DETECTION},
                ],
            })
            
            # Extract results
            labels = [label.description for label in response.label_annotations]
            texts = [text.description for text in response.text_annotations]
            objects = [obj.name for obj in response.localized_object_annotations]
            
            return {
                "labels": labels,
                "texts": texts,
                "objects": objects,
                "has_faces": len(response.face_annotations) > 0
            }
        
        # Run the synchronous API call in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_sync)
        logger.info(f"🔍 Vision API analysis complete: {len(result.get('labels', []))} labels, {len(result.get('objects', []))} objects")
        return result
        
    except Exception as e:
        logger.error(f"❌ Vision API error: {e}")
        return {"error": str(e)}

async def generate_answer(vision_data: dict, question: str) -> str:
    """Generate answer based on vision analysis and question"""
    labels = vision_data.get("labels", [])
    texts = vision_data.get("texts", [])
    objects = vision_data.get("objects", [])
    
    # Enhanced answer generation
    if vision_data.get("error"):
        return f"I encountered an error analyzing the image: {vision_data['error']}"
    
    base_info = f"Based on the image analysis, I can see: {', '.join(labels[:3])}" if labels else "I can analyze this image"
    
    if question.lower() in ['what is this', 'what do you see', 'describe']:
        if objects:
            answer = f"{base_info}. This appears to be an image containing {', '.join(objects[:3])}."
        else:
            answer = f"{base_info}. I can identify: {', '.join(labels[:5])}."
    elif 'text' in question.lower() or 'read' in question.lower():
        if texts:
            answer = f"I found text in the image: {texts[0][:200]}..."
        else:
            answer = "I couldn't detect any readable text in this image."
    elif 'count' in question.lower() or 'how many' in question.lower():
        answer = f"I can identify {len(objects)} distinct objects: {', '.join(objects)}." if objects else "I can see several elements but cannot provide an exact count."
    else:
        answer = f"{base_info}. I can provide more specific information if you ask a more detailed question about what you're looking for."
    
    return answer

# WebSocket connection info endpoint
@app.get("/ws-info")
async def websocket_info():
    """Get WebSocket connection information"""
    return {
        "active_connections": len(manager.active_connections),
        "endpoint": "/ws/{client_id}",
        "supported_message_types": [
            {
                "type": "analyze_image",
                "description": "Analyze an image with optional question",
                "required_fields": ["image"],
                "optional_fields": ["question"]
            },
            {
                "type": "ping",
                "description": "Ping the server",
                "required_fields": [],
                "optional_fields": ["timestamp"]
            }
        ],
        "response_types": [
            "connection_confirmed",
            "status",
            "vision_analysis", 
            "analysis_complete",
            "pong",
            "error"
        ]
    }