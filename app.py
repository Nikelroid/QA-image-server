# app.py - Improved version with better question-answer matching
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import json
import base64
import uuid
import asyncio
import os
import logging
import sys
import re
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

# Question categorization and processing
class QuestionAnalyzer:
    def __init__(self):
        self.question_patterns = {
            'count': [r'\bhow many\b', r'\bcount\b', r'\bnumber of\b', r'\bhow much\b'],
            'identify': [r'\bwhat is\b', r'\bwhat are\b', r'\bidentify\b', r'\bname\b', r'\btell me about\b'],
            'describe': [r'\bdescribe\b', r'\bexplain\b', r'\bdetails\b', r'\btell me\b'],
            'find': [r'\bfind\b', r'\blocate\b', r'\bwhere\b', r'\bshow me\b', r'\bpoint out\b'],
            'read_text': [r'\bread\b', r'\btext\b', r'\bwords\b', r'\bwriting\b', r'\bsays\b'],
            'color': [r'\bcolor\b', r'\bcolour\b', r'\bshade\b', r'\bhue\b'],
            'size': [r'\bsize\b', r'\bbig\b', r'\bsmall\b', r'\blarge\b', r'\btiny\b', r'\bhuge\b'],
            'location': [r'\bwhere\b', r'\bposition\b', r'\blocation\b', r'\bplace\b'],
            'comparison': [r'\bcompare\b', r'\bdifference\b', r'\bsimilar\b', r'\bbetter\b', r'\bworse\b'],
            'yes_no': [r'\bis there\b', r'\bcan you see\b', r'\bdoes it have\b', r'\bis it\b']
        }
    
    def categorize_question(self, question: str) -> str:
        question_lower = question.lower()
        
        for category, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return category
        
        return 'general'
    
    def extract_keywords(self, question: str) -> List[str]:
        # Remove common words and extract meaningful keywords
        stop_words = {'what', 'is', 'are', 'the', 'this', 'that', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords

question_analyzer = QuestionAnalyzer()

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
            "Smart question-answer matching",
            "Object detection and identification",
            "Text recognition and reading",
            "Real-time WebSocket communication",
            "Cloud Storage integration"
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
        answer = await generate_smart_answer(vision_results, request.question)
        
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
                    "features": ["smart_image_analysis", "question_matching", "real_time_processing"]
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
            "data": {"message": "Analyzing image...", "status": "processing"}
        }, client_id)
        
        image_base64 = data.get("image")
        question = data.get("question", "What do you see in this image?")
        
        if not image_base64:
            raise ValueError("No image data provided")
        
        logger.info(f"🔍 WebSocket image analysis for client {client_id}: {question}")
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Upload image to Cloud Storage (optional for WebSocket)
        upload_task = asyncio.create_task(upload_image_to_storage(image_data))
        
        # Analyze image with Vision API
        vision_results = await analyze_with_vision_api(image_data)
        
        # Send intermediate results
        await manager.send_personal_message({
            "type": "vision_analysis",
            "data": {"message": "Generating answer...", "status": "answering"}
        }, client_id)
        
        # Generate smart answer
        answer = await generate_smart_answer(vision_results, question)
        
        # Wait for upload to complete
        image_url = await upload_task
        
        # Send final result
        await manager.send_personal_message({
            "type": "analysis_complete",
            "data": {
                "answer": answer,
                "image_url": image_url,
                "question": question
            }
        }, client_id)
        
    except Exception as e:
        logger.error(f"❌ WebSocket analysis error: {e}")
        await manager.send_personal_message({
            "type": "error",
            "data": {"message": f"Analysis failed: {str(e)}"}
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
        logger.info(f"📤 Image uploaded successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error uploading image: {e}")
        return None

async def analyze_with_vision_api(image_data: bytes) -> dict:
    """Analyze image using Google Cloud Vision API with comprehensive features"""
    try:
        if not vision_client:
            return {"error": "Vision client not initialized"}
            
        def analyze_sync():
            image = vision.Image(content=image_data)
            
            # Perform comprehensive analysis
            response = vision_client.annotate_image({
                'image': image,
                'features': [
                    {'type_': vision.Feature.Type.LABEL_DETECTION, 'max_results': 20},
                    {'type_': vision.Feature.Type.TEXT_DETECTION},
                    {'type_': vision.Feature.Type.OBJECT_LOCALIZATION, 'max_results': 20},
                    {'type_': vision.Feature.Type.FACE_DETECTION, 'max_results': 10},
                    {'type_': vision.Feature.Type.LANDMARK_DETECTION, 'max_results': 10},
                    {'type_': vision.Feature.Type.LOGO_DETECTION, 'max_results': 10},
                    {'type_': vision.Feature.Type.IMAGE_PROPERTIES},
                ],
            })
            
            # Extract comprehensive results
            labels = [(label.description, label.score) for label in response.label_annotations]
            
            # Extract text with positions
            texts = []
            if response.text_annotations:
                # First annotation contains full text
                full_text = response.text_annotations[0].description if response.text_annotations else ""
                for text_annotation in response.text_annotations[1:]:  # Skip the first one (full text)
                    texts.append({
                        'text': text_annotation.description,
                        'confidence': getattr(text_annotation, 'confidence', 0.0)
                    })
            else:
                full_text = ""
            
            # Extract objects with positions and confidence
            objects = []
            for obj in response.localized_object_annotations:
                objects.append({
                    'name': obj.name,
                    'confidence': obj.score,
                    'bounding_box': {
                        'vertices': [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
                    }
                })
            
            # Extract faces
            faces = []
            for face in response.face_annotations:
                faces.append({
                    'confidence': face.detection_confidence,
                    'joy': face.joy_likelihood.name,
                    'anger': face.anger_likelihood.name,
                    'surprise': face.surprise_likelihood.name
                })
            
            # Extract landmarks
            landmarks = [landmark.description for landmark in response.landmark_annotations]
            
            # Extract logos
            logos = [logo.description for logo in response.logo_annotations]
            
            # Extract colors
            colors = []
            if response.image_properties_annotation and response.image_properties_annotation.dominant_colors:
                for color in response.image_properties_annotation.dominant_colors.colors[:5]:
                    colors.append({
                        'red': color.color.red,
                        'green': color.color.green,
                        'blue': color.color.blue,
                        'score': color.score
                    })
            
            return {
                "labels": labels,
                "full_text": full_text,
                "texts": texts,
                "objects": objects,
                "faces": faces,
                "landmarks": landmarks,
                "logos": logos,
                "dominant_colors": colors,
                "has_faces": len(faces) > 0,
                "has_text": bool(full_text.strip()),
                "object_count": len(objects)
            }
        
        # Run the synchronous API call in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_sync)
        logger.info(f"🔍 Vision API analysis complete: {len(result.get('labels', []))} labels, {len(result.get('objects', []))} objects")
        return result
        
    except Exception as e:
        logger.error(f"❌ Vision API error: {e}")
        return {"error": str(e)}

async def generate_smart_answer(vision_data: dict, question: str) -> str:
    """Generate intelligent answer based on vision analysis and question context"""
    
    if vision_data.get("error"):
        return f"I encountered an error analyzing the image: {vision_data['error']}"
    
    # Categorize the question
    question_category = question_analyzer.categorize_question(question)
    keywords = question_analyzer.extract_keywords(question)
    
    logger.info(f"Question category: {question_category}, Keywords: {keywords}")
    
    # Extract data from vision results
    labels = vision_data.get("labels", [])
    objects = vision_data.get("objects", [])
    full_text = vision_data.get("full_text", "")
    faces = vision_data.get("faces", [])
    landmarks = vision_data.get("landmarks", [])
    logos = vision_data.get("logos", [])
    colors = vision_data.get("dominant_colors", [])
    
    # Generate category-specific answers
    if question_category == "count":
        return generate_count_answer(objects, labels, keywords, question)
    
    elif question_category == "read_text":
        if full_text.strip():
            return f"The text in the image reads: '{full_text.strip()}'"
        else:
            return "I cannot detect any readable text in this image."
    
    elif question_category == "identify":
        return generate_identify_answer(objects, labels, landmarks, logos, keywords, question)
    
    elif question_category == "find":
        return generate_find_answer(objects, labels, keywords, question)
    
    elif question_category == "color":
        return generate_color_answer(colors, objects, labels, keywords)
    
    elif question_category == "yes_no":
        return generate_yes_no_answer(objects, labels, full_text, faces, keywords, question)
    
    elif question_category == "describe":
        return generate_description_answer(objects, labels, faces, colors, landmarks, keywords)
    
    elif question_category == "location":
        return generate_location_answer(objects, landmarks, keywords)
    
    else:
        # General question handling
        return generate_general_answer(objects, labels, full_text, faces, landmarks, logos, keywords, question)

def generate_count_answer(objects, labels, keywords, question):
    """Generate answer for counting questions"""
    # Look for specific objects to count
    for keyword in keywords:
        matching_objects = [obj for obj in objects if keyword.lower() in obj['name'].lower()]
        if matching_objects:
            return f"I can see {len(matching_objects)} {keyword}(s) in the image."
    
    # Count all objects if no specific keyword found
    if objects:
        object_counts = {}
        for obj in objects:
            name = obj['name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        count_text = ", ".join([f"{count} {name}" for name, count in object_counts.items()])
        return f"I can count these objects: {count_text}."
    
    return f"I can identify {len(labels)} different elements in the image, but cannot provide specific counts."

def generate_identify_answer(objects, labels, landmarks, logos, keywords, question):
    """Generate answer for identification questions"""
    # Check for specific keywords in objects first
    for keyword in keywords:
        matching_objects = [obj for obj in objects if keyword.lower() in obj['name'].lower()]
        if matching_objects:
            obj = matching_objects[0]
            return f"I can see a {obj['name']} in the image (confidence: {obj['confidence']:.2f})."
    
    # Check landmarks
    if landmarks:
        return f"This appears to be {landmarks[0]}."
    
    # Check logos
    if logos:
        return f"I can identify the {logos[0]} logo in this image."
    
    # Use most confident objects or labels
    if objects:
        main_object = objects[0]
        return f"The main subject appears to be a {main_object['name']} (confidence: {main_object['confidence']:.2f})."
    
    if labels:
        top_labels = [label[0] for label in labels[:3]]
        return f"This image contains: {', '.join(top_labels)}."
    
    return "I can see various elements in the image, but need more specific details to identify what you're looking for."

def generate_find_answer(objects, labels, keywords, question):
    """Generate answer for finding/locating questions"""
    found_items = []
    
    for keyword in keywords:
        # Check in objects
        matching_objects = [obj for obj in objects if keyword.lower() in obj['name'].lower()]
        if matching_objects:
            found_items.append(f"I found {keyword} in the image")
        
        # Check in labels
        matching_labels = [label for label in labels if keyword.lower() in label[0].lower()]
        if matching_labels and not matching_objects:
            found_items.append(f"I can see {keyword} in the image")
    
    if found_items:
        return ". ".join(found_items) + "."
    
    return f"I cannot locate the specific item you're looking for. I can see: {', '.join([obj['name'] for obj in objects[:3]])}."

def generate_color_answer(colors, objects, labels, keywords):
    """Generate answer for color-related questions"""
    if not colors:
        return "I cannot determine the dominant colors in this image."
    
    # Convert RGB to color names (simplified)
    def rgb_to_color_name(r, g, b):
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "red" if r > 150 else "dark red"
        elif g > r and g > b:
            return "green" if g > 150 else "dark green"
        elif b > r and b > g:
            return "blue" if b > 150 else "dark blue"
        elif r > 150 and g > 150:
            return "yellow"
        elif r > 150 and b > 150:
            return "purple"
        elif g > 150 and b > 150:
            return "cyan"
        else:
            return "brown"
    
    main_color = colors[0]
    color_name = rgb_to_color_name(main_color['red'], main_color['green'], main_color['blue'])
    
    return f"The dominant color in this image is {color_name}."

def generate_yes_no_answer(objects, labels, full_text, faces, keywords, question):
    """Generate yes/no answers"""
    question_lower = question.lower()
    
    # Check for faces
    if "face" in keywords or "person" in keywords:
        if faces:
            return f"Yes, I can see {len(faces)} face(s) in the image."
        else:
            return "No, I cannot see any faces in this image."
    
    # Check for text
    if "text" in keywords or "writing" in keywords:
        if full_text.strip():
            return "Yes, there is text in this image."
        else:
            return "No, I cannot see any text in this image."
    
    # Check for specific objects
    for keyword in keywords:
        matching_objects = [obj for obj in objects if keyword.lower() in obj['name'].lower()]
        matching_labels = [label for label in labels if keyword.lower() in label[0].lower()]
        
        if matching_objects or matching_labels:
            return f"Yes, I can see {keyword} in the image."
    
    return "I need more specific information to answer your yes/no question."

def generate_description_answer(objects, labels, faces, colors, landmarks, keywords):
    """Generate descriptive answers"""
    description_parts = []
    
    # Main subjects
    if objects:
        main_objects = [obj['name'] for obj in objects[:3]]
        description_parts.append(f"This image shows {', '.join(main_objects)}")
    
    # People
    if faces:
        description_parts.append(f"with {len(faces)} person(s)")
    
    # Landmarks
    if landmarks:
        description_parts.append(f"featuring {landmarks[0]}")
    
    # Colors
    if colors and len(colors) >= 2:
        color1 = rgb_to_color_name(colors[0]['red'], colors[0]['green'], colors[0]['blue'])
        color2 = rgb_to_color_name(colors[1]['red'], colors[1]['green'], colors[1]['blue'])
        description_parts.append(f"with {color1} and {color2} as dominant colors")
    
    if description_parts:
        return ". ".join(description_parts) + "."
    
    return "This image contains various visual elements that I can analyze if you ask more specific questions."

def generate_location_answer(objects, landmarks, keywords):
    """Generate location-based answers"""
    if landmarks:
        return f"This appears to be taken at or near {landmarks[0]}."
    
    # Look for location indicators in objects
    location_objects = [obj for obj in objects if any(loc in obj['name'].lower() 
                       for loc in ['building', 'street', 'park', 'beach', 'mountain', 'city', 'room', 'kitchen', 'office'])]
    
    if location_objects:
        return f"This appears to be in or around a {location_objects[0]['name']}."
    
    return "I cannot determine the specific location from this image."

def generate_general_answer(objects, labels, full_text, faces, landmarks, logos, keywords, question):
    """Generate general answers for uncategorized questions"""
    
    # Try to match keywords with detected content
    relevant_info = []
    
    # Check objects for keyword matches
    for keyword in keywords:
        matching_objects = [obj for obj in objects if keyword.lower() in obj['name'].lower()]
        if matching_objects:
            relevant_info.append(f"{matching_objects[0]['name']} (detected with {matching_objects[0]['confidence']:.1%} confidence)")
    
    # Check labels for keyword matches
    if not relevant_info:
        for keyword in keywords:
            matching_labels = [label for label in labels if keyword.lower() in label[0].lower() and label[1] > 0.5]
            if matching_labels:
                relevant_info.append(f"{matching_labels[0][0]} (confidence: {matching_labels[0][1]:.1%})")
    
    # If we found relevant matches, return focused answer
    if relevant_info:
        return f"Regarding your question about the image: I can see {', '.join(relevant_info)}."
    
    # Fallback to most confident detections
    if objects:
        top_object = objects[0]
        return f"In this image, I can clearly identify a {top_object['name']} (confidence: {top_object['confidence']:.1%})."
    
    if labels:
        top_labels = [label[0] for label in labels[:2] if label[1] > 0.7]
        if top_labels:
            return f"This image appears to contain: {', '.join(top_labels)}."
    
    return "I can see various elements in this image. Could you ask a more specific question about what you'd like to know?"

def rgb_to_color_name(r, g, b):
    """Convert RGB values to color names"""
    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > g and r > b:
        return "red" if r > 150 else "dark red"
    elif g > r and g > b:
        return "green" if g > 150 else "dark green"
    elif b > r and b > g:
        return "blue" if b > 150 else "dark blue"
    elif r > 150 and g > 150:
        return "yellow"
    elif r > 150 and b > 150:
        return "purple"
    elif g > 150 and b > 150:
        return "cyan"
    else:
        return "brown"

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
                "description": "Analyze an image with intelligent question matching",
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
        ],
        "question_categories": [
            "count - How many objects, counting questions",
            "identify - What is this, identification questions", 
            "describe - Describe, explain, tell me about",
            "find - Find, locate, where is questions",
            "read_text - Read text, what does it say",
            "color - Color related questions",
            "yes_no - Yes/no questions (is there, can you see)",
            "location - Where, position, place questions",
            "general - All other questions"
        ]
    }