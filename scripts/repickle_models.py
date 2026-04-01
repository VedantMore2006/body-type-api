import joblib
import os
import sys
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def repickle_models():
    model_paths = [
        "bodytype_model.pkl",
        "label_encoder.pkl"
    ]
    
    print(f"Current sklearn version: {sklearn.__version__}")
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        print(f"Processing {path}...")
        try:
            # We use joblib because the main code uses joblib.load
            obj = joblib.load(path)
                
            # Re-save the object in the current environment
            joblib.dump(obj, path)
            print(f"Successfully re-pickled {path} using joblib")
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    repickle_models()
