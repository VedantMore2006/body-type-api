# Body Type & Dosha Prediction API

This project contains a high-performance, image-based machine learning pipeline structured as a **FastAPI** web service. The API analyzes user-submitted frontal images along with their basic physical information (age, gender, weight, height) to assess their physical profile out of Western Body Types (Ectomorph, Mesomorph, Endomorph) and their equivalent Ayurvedic Dosha types (Vata, Pitta, Kapha).

## Features

- **Fast & Optimized**: Leverages a single `Lifespan` application boot cycle to load large ML models (YOLO, scikit-learn models) only once into active memory.
- **Dynamic Geometric Scaling**: Accurately computes body ratios directly from user pixel space mapped against true dimensions dynamically. 
- **Graceful Error Handling**: Automatically senses and rejects inherently flawed images (e.g., if a body is cut off arbitrarily, or if conditions are too brightly lit or dark) and handles them via clean `400 Bad Request` signals without crashing.
- **Ayurvedic Integration**: Seamless crossover from standard physical geometry mapping into classical Dosha traits.

---

## Technical Stack
- **Framework**: `FastAPI`
- **Web Server**: `Uvicorn`
- **Vision ML Models**: `Ultralytics YOLOv8` (Pose and Detection)
- **Image Processing**: `OpenCV`, `MediaPipe`
- **Classification Modeling**: `Scikit-Learn`, `Joblib`

---

## Directory Structure

```text
body-type-api/
│
├── app.py                      # Main entrypoint housing the FastAPI service routines
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation file
├── bodytype_model.pkl          # Scikit-Learn Model Binaries
├── label_encoder.pkl           # Label Encoder Definitions
├── yolov8n.pt                  # YOLO Base Detection Model
├── yolov8n-pose.pt             # YOLO Pose Estimation Model
│
├── measurement/                # Module for computing anatomical scaling and pixel ratios
├── pipeline/                   # Hub joining ML extractions with the math validations
├── models/                     # Orchestrating logic handling ML model loads
├── vision/                     # Lower level YOLO/CV bindings for cropping and detection
└── extra/                      # Contains scratch scripts, datasets, backups, and previous pipeline iteration code (not loaded by API)
```

---

## Local Setup & Installation

### 1. Requirements

Ensure you are using `Python 3.10+`.

```bash
# Clone and enter directory
cd body-type-api

# Install required dependencies
pip install -r requirements.txt
```

### 2. Running the API

Start the web server asynchronously via Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
*The server will take several seconds on boot as it mounts the YOLO neural nets and classifiers into memory.*

---

## Using the API

The primary endpoint is documented out of the box interactively.

### Interactive Docs
Once running, just navigate to:
**[http://localhost:8000/docs](http://localhost:8000/docs)** 

### Endpoint: `POST /predict`

*Accepts explicitly formatted `multipart/form-data` uploads.*

**Form Parameters:**
- `image` (File): Input picture of the full subject standing frontally. Must be clear vertically.
- `age` (Float): e.g., 25
- `gender` (Float): `0` = Female, `1` = Male
- `person_height_cm` (Float): Actual height in cm e.g., 180 
- `person_weight_kg` (Float): Actual weight in kilogram e.g., 75

**Example Response (200 OK):**
```json
{
  "body_type": "Endomorph",
  "ayurvedic_type": "Kapha"
}
```

**Example Response (400 Bad Request - Geometric issues):**
```json
{
  "detail": "Person is not fully visible. Please capture full body in frame"
}
```

**Example Response (400 Bad Request - Lighting issues):**
```json
{
  "detail": "Lighting condition is not proper. The image is either too dark or too bright."
}
```

---

## Troubleshooting

- **Server starts lag/crash:** Please ensure you have sufficient RAM (minimum 2GB+) to comfortably cache the vision models initially.
- **Port already in use:** Verify another instance of the uvicorn API isn't running in the background. Use `killall python3` to flush hanging instances.
