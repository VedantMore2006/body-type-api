import io
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from models.model_loader import ModelLoader
from pipeline.measurement_pipeline import MeasurementPipeline


# Global dictionary for models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models on startup
    print("Loading models...")
    loader = ModelLoader()
    models = loader.load_models()
    ml_models["models"] = models
    print("Models loaded successfully.")
    yield
    # Clean up models on shutdown
    ml_models.clear()

app = FastAPI(title="Body Type API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": "models" in ml_models}

def check_lighting(image: np.ndarray) -> bool:
    """Check if the image is too dark or too bright."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 30: # Too dark
        return False
    if mean_brightness > 230: # Too bright
        return False
    return True

@app.post("/predict")
async def predict_body_type(
    image: UploadFile = File(...),
    age: float = Form(...),
    gender: float = Form(...),
    person_height_cm: float = Form(...),
    person_weight_kg: float = Form(...),
):
    try:
        # Read the uploaded file into a byte array
        contents = await image.read()
        
        # Decode the image via OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            raise ValueError("Could not decode the provided image. Please upload a valid image file.")

        # Lighting check
        if not check_lighting(img_cv2):
            raise ValueError("Lighting condition is not proper. The image is either too dark or too bright.")

        # Initialize the pipeline with the pre-loaded models
        pipeline = MeasurementPipeline(ml_models["models"])

        # Run pipeline
        # The pipeline handles "person is not properly visible" via validate_person_geometry which raises ValueError
        result = pipeline.run(
            image=img_cv2,
            age=age,
            gender=gender,
            person_height_cm=person_height_cm,
            person_weight_kg=person_weight_kg,
        )

        return JSONResponse(content=result)

    except ValueError as e:
        # Catch explicit value errors from pipeline (e.g. person not fully visible, decoding failed)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected errors
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing the image.")
