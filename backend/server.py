from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import re
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class CodeAnalysisRequest(BaseModel):
    code: str
    framework: Optional[str] = None

class NodePosition(BaseModel):
    x: float
    y: float

class LayerInfo(BaseModel):
    id: str
    name: str
    type: str
    params: Dict[str, Any]
    description: Optional[str] = None
    position: Optional[NodePosition] = None
    layer_index: int
    neurons: Optional[int] = None

class ConnectionInfo(BaseModel):
    source: str
    target: str
    weight: Optional[float] = None

class CodeAnalysisResponse(BaseModel):
    success: bool
    framework: Optional[str] = None
    model_type: Optional[str] = None
    layers: List[LayerInfo]
    connections: List[ConnectionInfo]
    architecture_summary: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = []

class VisualizationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    code: str
    framework: str
    layers: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ML Code Parser with Gemini AI
class GeminiMLCodeParser:
    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('EMERGENT_LLM_KEY')
        self.framework = None
        self.warnings = []
    
    async def parse(self, code: str) -> Dict[str, Any]:
        """Main parsing function using Gemini AI"""
        try:
            # Detect framework
            self.framework = self._detect_framework(code)
            
            if not self.framework:
                return {
                    'success': False,
                    'error': 'Unable to detect ML/DL framework. Please use TensorFlow/Keras, PyTorch, or Scikit-learn.',
                    'framework': None,
                    'model_type': None,
                    'layers': [],
                    'connections': [],
                    'architecture_summary': None,
                    'warnings': []
                }
            
            # Use Gemini to analyze the code
            analysis = await self._analyze_with_gemini(code, self.framework)
            return analysis
            
        except Exception as e:
            logging.error(f"Parsing error: {str(e)}")
            return {
                'success': False,
                'error': f'Analysis Error: {str(e)}',
                'framework': self.framework,
                'model_type': None,
                'layers': [],
                'connections': [],
                'architecture_summary': None,
                'warnings': self.warnings
            }
    
    def _detect_framework(self, code: str) -> Optional[str]:
        """Detect ML framework from code"""
        code_lower = code.lower()
        if 'tensorflow' in code_lower or 'keras' in code_lower:
            return 'Keras/TensorFlow'
        elif 'torch' in code_lower or 'nn.module' in code_lower:
            return 'PyTorch'
        elif 'sklearn' in code_lower or 'from sklearn' in code:
            return 'Scikit-learn'
        return None
    
    async def _analyze_with_gemini(self, code: str, framework: str) -> Dict[str, Any]:
        """Use Gemini AI to analyze ML/DL code structure"""
        
        system_prompt = """You are an expert ML/DL code analyzer. Analyze the provided code and extract the neural network architecture.
        
Return your analysis in this EXACT JSON format (no markdown, no code blocks, just pure JSON):
{
  "success": true,
  "framework": "framework name",
  "model_type": "sequential/functional/custom",
  "architecture_summary": "brief description of the architecture",
  "layers": [
    {
      "id": "layer_0",
      "name": "Layer Name",
      "type": "Dense/Conv2d/LSTM/etc",
      "layer_index": 0,
      "neurons": number_of_neurons_or_units,
      "params": {"key": "value"},
      "description": "what this layer does"
    }
  ],
  "connections": [
    {"source": "layer_0", "target": "layer_1"}
  ],
  "warnings": []
}

Rules:
1. Extract ALL layers including input, hidden, and output layers
2. For neurons: use the units/features parameter (e.g., Dense(128) has 128 neurons)
3. Create sequential connections between layers
4. Include activation functions, dropout rates, kernel sizes in params
5. Provide clear descriptions for each layer
6. Return ONLY valid JSON, no extra text"""
        
        user_prompt = f"""Analyze this {framework} code and extract the neural network architecture:

```python
{code}
```

Return the analysis in the specified JSON format."""
        
        try:
            # Initialize Gemini chat
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"code-analysis-{uuid.uuid4()}",
                system_message=system_prompt
            ).with_model("gemini", "gemini-2.0-flash")
            
            # Send message and get response
            user_message = UserMessage(text=user_prompt)
            response = await chat.send_message(user_message)
            
            # Parse JSON response
            import json
            
            # Clean response (remove markdown code blocks if present)
            response_text = response.strip()
            if response_text.startswith('```'):
                # Extract JSON from markdown code block
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(response_text)
            
            # Add positions for circular layout
            result = self._add_node_positions(result)
            
            return result
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.error(f"Response was: {response[:500]}")
            # Fallback to basic parsing
            return self._fallback_parse(code, framework)
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return self._fallback_parse(code, framework)
    
    def _add_node_positions(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate positions for circular node layout"""
        layers = result.get('layers', [])
        num_layers = len(layers)
        
        if num_layers == 0:
            return result
        
        # Calculate positions for layers (left to right)
        layer_spacing = 800 / (num_layers + 1) if num_layers > 1 else 400
        
        for idx, layer in enumerate(layers):
            x_pos = 100 + (idx + 1) * layer_spacing
            y_pos = 250  # Center Y position
            
            layer['position'] = {
                'x': x_pos,
                'y': y_pos
            }
        
        return result
    
    def _fallback_parse(self, code: str, framework: str) -> Dict[str, Any]:
        """Fallback parsing when Gemini fails"""
        layers = self._create_generic_structure(framework)
        connections = []
        
        for i in range(len(layers) - 1):
            connections.append({
                'source': layers[i]['id'],
                'target': layers[i + 1]['id']
            })
        
        result = {
            'success': True,
            'framework': framework,
            'model_type': 'sequential',
            'layers': layers,
            'connections': connections,
            'architecture_summary': 'Generic neural network structure (unable to parse specific layers)',
            'warnings': ['Using fallback parser - could not analyze code with AI']
        }
        
        return self._add_node_positions(result)
    
    def _create_generic_structure(self, framework: str) -> List[Dict[str, Any]]:
        """Create a generic neural network structure"""
        if framework == 'Scikit-learn':
            return [
                {
                    'id': 'layer_0',
                    'name': 'Input Features',
                    'type': 'Input',
                    'layer_index': 0,
                    'neurons': 10,
                    'params': {'description': 'Input data'},
                    'description': 'Training data features'
                },
                {
                    'id': 'layer_1',
                    'name': 'ML Algorithm',
                    'type': 'Algorithm',
                    'layer_index': 1,
                    'neurons': 1,
                    'params': {},
                    'description': 'Machine learning model'
                },
                {
                    'id': 'layer_2',
                    'name': 'Predictions',
                    'type': 'Output',
                    'layer_index': 2,
                    'neurons': 1,
                    'params': {},
                    'description': 'Model predictions'
                }
            ]
        else:
            return [
                {
                    'id': 'layer_0',
                    'name': 'Input Layer',
                    'type': 'Input',
                    'layer_index': 0,
                    'neurons': 128,
                    'params': {},
                    'description': 'Input features to the network'
                },
                {
                    'id': 'layer_1',
                    'name': 'Hidden Layer 1',
                    'type': 'Dense',
                    'layer_index': 1,
                    'neurons': 64,
                    'params': {'activation': 'relu'},
                    'description': 'First hidden layer with ReLU activation'
                },
                {
                    'id': 'layer_2',
                    'name': 'Hidden Layer 2',
                    'type': 'Dense',
                    'layer_index': 2,
                    'neurons': 32,
                    'params': {'activation': 'relu'},
                    'description': 'Second hidden layer with ReLU activation'
                },
                {
                    'id': 'layer_3',
                    'name': 'Output Layer',
                    'type': 'Output',
                    'layer_index': 3,
                    'neurons': 10,
                    'params': {'activation': 'softmax'},
                    'description': 'Output layer for classification'
                }
            ]


# Routes
@api_router.get("/")
async def root():
    return {"message": "Neural Canvas API", "status": "active", "ai_powered": True}


@api_router.post("/analyze-code", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """
    Analyze ML/DL code using Gemini AI and extract model structure
    """
    try:
        parser = GeminiMLCodeParser()
        result = await parser.parse(request.code)
        return result
    except Exception as e:
        logging.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/save-visualization")
async def save_visualization(record: VisualizationRecord):
    """
    Save visualization to database
    """
    try:
        doc = record.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.visualizations.insert_one(doc)
        return {"success": True, "id": record.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/visualizations")
async def get_visualizations():
    """
    Get all saved visualizations
    """
    try:
        visualizations = await db.visualizations.find({}, {"_id": 0}).to_list(1000)
        for viz in visualizations:
            if isinstance(viz['timestamp'], str):
                viz['timestamp'] = datetime.fromisoformat(viz['timestamp'])
        return {"success": True, "visualizations": visualizations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()