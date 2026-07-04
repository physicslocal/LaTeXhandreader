import asyncio
import json
import logging
import time
import base64
import io
from typing import List, Dict, Any
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MathToLatex")

class AppConfig:
    HOST = "0.0.0.0"
    PORT = 8000
    WS_PATH = "/ws"
    MAX_IMAGE_SIZE = (672, 192)

config = AppConfig()

# ==========================================
# IMAGE PROCESSING
# ==========================================
class ImageProcessor:
    @staticmethod
    def base64_to_pil(base64_str: str) -> Image.Image:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data)).convert("L")

    @staticmethod
    def preprocess_for_model(img: Image.Image) -> Image.Image:
        img = ImageOps.invert(img)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        img = ImageOps.expand(img, border=20, fill='white')
        img.thumbnail(config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        new_img = Image.new('L', config.MAX_IMAGE_SIZE, 'white')
        new_img.paste(img, (0, 0))
        return new_img

# ==========================================
# INFERENCE ENGINE (MOCK FOR DEPLOYMENT)
# ==========================================
class MathRecognizer:
    async def recognize(self, base64_img: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            raw_img = ImageProcessor.base64_to_pil(base64_img)
            processed_img = ImageProcessor.preprocess_for_model(raw_img)
            
            # หน่วงเวลาจำลอง GPU ตามสเปก
            await asyncio.sleep(0.15) 
            
            stat = ImageOps.stat(processed_img)
            ink_density = stat.mean[0] if isinstance(stat.mean, list) else stat.mean
            
            latency = int((time.time() - start_time) * 1000)
            
            if ink_density < 50:
                latex, conf = "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", 0.98
            elif ink_density < 100:
                latex, conf = "\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}", 0.94
            elif ink_density < 150:
                latex, conf = "\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}", 0.89
            else:
                latex, conf = "\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}", 0.95

            return {"latex": latex, "confidence": conf, "latency_ms": latency, "status": "success"}
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return {"latex": "", "confidence": 0, "latency_ms": 0, "status": "error", "message": str(e)}

recognizer = MathRecognizer()

# ==========================================
# FASTAPI APP & CORS
# ==========================================
app = FastAPI(title="Math to LaTeX Backend API")

# ⚠️ การตั้งค่า CORS (อนุญาตให้ GitHub Pages ยิงเข้ามาได้)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ตอนรันจริงสามารถเปลี่ยนเป็น ["https://username.github.io"] ได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# WEBSOCKET MANAGER
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.websocket(config.WS_PATH)
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
                
            if "image" in payload:
                result = await recognizer.recognize(payload["image"])
                await websocket.send_json({"type": "result", "data": result})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return {"status": "Backend is running! Ready for WebSocket connections."}

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
