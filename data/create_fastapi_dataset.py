"""
FastAPI Specialist Dataset Creator

This script creates a high-quality dataset for fine-tuning CodeLlama 
to become a FastAPI specialist by mining GitHub repositories and 
creating instruction-response pairs.
"""

import json
import requests
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from urllib.parse import urljoin


@dataclass
class FastAPIExample:
    """Structure for a FastAPI code example."""
    instruction: str
    input: str
    output: str
    category: str
    difficulty: str
    tags: List[str]


class FastAPIDatasetCreator:
    """Creates specialized FastAPI training dataset."""
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the dataset creator.
        
        Args:
            github_token: GitHub API token for higher rate limits
        """
        self.github_token = github_token
        self.headers = {
            "Authorization": f"token {github_token}" if github_token else None,
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"
        self.examples = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def search_fastapi_repos(self, max_repos: int = 50) -> List[Dict]:
        """Search for high-quality FastAPI repositories."""
        search_queries = [
            "fastapi stars:>100 language:python",
            "fastapi tutorial language:python",
            "fastapi example language:python",
            "fastapi rest api language:python"
        ]
        
        repos = []
        for query in search_queries:
            url = f"{self.base_url}/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "per_page": max_repos // len(search_queries)
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                repos.extend(data.get("items", []))
                time.sleep(1)  # Rate limiting
                
        return repos[:max_repos]
        
    def extract_fastapi_files(self, repo_data: Dict) -> List[Dict]:
        """Extract Python files from a FastAPI repository."""
        files = []
        contents_url = repo_data["contents_url"].replace("{+path}", "")
        
        try:
            response = requests.get(contents_url, headers=self.headers)
            if response.status_code == 200:
                contents = response.json()
                
                for item in contents:
                    if item["type"] == "file" and item["name"].endswith(".py"):
                        if any(keyword in item["name"].lower() for keyword in 
                              ["main", "app", "api", "router", "endpoint"]):
                            files.append(item)
                            
        except Exception as e:
            self.logger.warning(f"Error extracting files from {repo_data['name']}: {e}")
            
        return files
        
    def download_file_content(self, file_url: str) -> Optional[str]:
        """Download file content from GitHub."""
        try:
            response = requests.get(file_url, headers=self.headers)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            self.logger.warning(f"Error downloading file: {e}")
            
        return None
        
    def create_basic_examples(self) -> List[FastAPIExample]:
        """Create basic FastAPI examples covering fundamental concepts."""
        examples = []
        
        # Basic FastAPI app
        examples.append(FastAPIExample(
            instruction="Create a simple FastAPI application with a hello world endpoint",
            input="",
            output="""from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}""",
            category="basic",
            difficulty="beginner",
            tags=["fastapi", "basic", "endpoint"]
        ))
        
        # Pydantic model
        examples.append(FastAPIExample(
            instruction="Create a FastAPI endpoint that accepts a user registration with email validation",
            input="",
            output="""from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from typing import Optional

app = FastAPI()

class UserRegistration(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None

@app.post("/register")
async def register_user(user: UserRegistration):
    return {
        "message": "User registered successfully",
        "username": user.username,
        "email": user.email
    }""",
            category="models",
            difficulty="intermediate",
            tags=["fastapi", "pydantic", "validation", "registration"]
        ))
        
        # Database integration
        examples.append(FastAPIExample(
            instruction="Create a FastAPI endpoint to get all users from database using SQLAlchemy",
            input="",
            output="""from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, User

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users""",
            category="database",
            difficulty="intermediate",
            tags=["fastapi", "sqlalchemy", "database", "dependency-injection"]
        ))
        
        # Authentication
        examples.append(FastAPIExample(
            instruction="Create a FastAPI endpoint with JWT authentication",
            input="",
            output="""from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/protected")
async def protected_route(current_user: str = Depends(verify_token)):
    return {"message": f"Hello {current_user}"}""",
            category="authentication",
            difficulty="advanced",
            tags=["fastapi", "jwt", "authentication", "security"]
        ))
        
        # File upload
        examples.append(FastAPIExample(
            instruction="Create a FastAPI endpoint to upload and process files",
            input="",
            output="""from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import shutil
from pathlib import Path

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.jpg', '.jpeg', '.png', '.pdf')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = Path(f"uploads/{file.filename}")
    file_path.parent.mkdir(exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "size": file_path.stat().st_size,
        "content_type": file.content_type
    }

@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    
    for file in files:
        file_path = Path(f"uploads/{file.filename}")
        file_path.parent.mkdir(exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_files.append({
            "filename": file.filename,
            "size": file_path.stat().st_size
        })
    
    return {"uploaded_files": uploaded_files}""",
            category="file-handling",
            difficulty="intermediate",
            tags=["fastapi", "file-upload", "multipart"]
        ))
        
        return examples
        
    def create_advanced_examples(self) -> List[FastAPIExample]:
        """Create advanced FastAPI examples."""
        examples = []
        
        # Background tasks
        examples.append(FastAPIExample(
            instruction="Create a FastAPI endpoint that sends email in background after user registration",
            input="",
            output="""from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, EmailStr
import smtplib
from email.mime.text import MIMEText

app = FastAPI()

class User(BaseModel):
    email: EmailStr
    name: str

def send_welcome_email(email: str, name: str):
    # Email sending logic
    msg = MIMEText(f"Welcome {name}!")
    msg['Subject'] = 'Welcome to our platform'
    msg['From'] = 'noreply@example.com'
    msg['To'] = email
    
    # Send email logic here
    print(f"Sending email to {email}")

@app.post("/register")
async def register_user(user: User, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_welcome_email, user.email, user.name)
    return {"message": "User registered successfully"}""",
            category="background-tasks",
            difficulty="advanced",
            tags=["fastapi", "background-tasks", "email"]
        ))
        
        # WebSocket
        examples.append(FastAPIExample(
            instruction="Create a FastAPI WebSocket endpoint for real-time chat",
            input="",
            output="""from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")""",
            category="websocket",
            difficulty="advanced",
            tags=["fastapi", "websocket", "real-time", "chat"]
        ))
        
        return examples
        
    def create_dataset(self, output_path: str = "data/fastapi_specialist.json"):
        """Create the complete FastAPI specialist dataset."""
        self.logger.info("Creating FastAPI specialist dataset...")
        
        # Add basic examples
        basic_examples = self.create_basic_examples()
        self.examples.extend(basic_examples)
        
        # Add advanced examples
        advanced_examples = self.create_advanced_examples()
        self.examples.extend(advanced_examples)
        
        # Convert to format compatible with your existing pipeline
        dataset = []
        for example in self.examples:
            dataset.append({
                "instruction": example.instruction,
                "input": example.input,
                "output": example.output,
                "category": example.category,
                "difficulty": example.difficulty,
                "tags": example.tags
            })
        
        # Save dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Dataset created with {len(dataset)} examples")
        self.logger.info(f"Saved to: {output_path}")
        
        # Print statistics
        categories = {}
        difficulties = {}
        
        for item in dataset:
            categories[item['category']] = categories.get(item['category'], 0) + 1
            difficulties[item['difficulty']] = difficulties.get(item['difficulty'], 0) + 1
        
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"Categories: {categories}")
        self.logger.info(f"Difficulties: {difficulties}")
        
        return dataset


if __name__ == "__main__":
    # Create the dataset
    creator = FastAPIDatasetCreator()
    dataset = creator.create_dataset()
    
    print(f"✅ FastAPI specialist dataset created with {len(dataset)} examples")
    print("🚀 Ready to fine-tune your specialized model!")