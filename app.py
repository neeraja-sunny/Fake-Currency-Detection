import os
# Suppress TensorFlow informational messages and oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Image dimensions required by the model
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Load main model globally
model = None
MODEL_PATH = 'currency_model.h5'

# Load general object detection model (MobileNetV2)
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    
    # Load MobileNetV2 without the top layers for general classification
    general_model = MobileNetV2(weights='imagenet')
    print("Loaded MobileNetV2 for general object detection.")
except Exception as e:
    print(f"Error loading MobileNetV2: {e}")
    general_model = None

def init_model():
    """Initializes and loads the pre-trained model."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
        else:
            print(f"Warning: {MODEL_PATH} not found. Please run train_model.py first.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Try to load model at startup
init_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocesses the image to match the model's expected input:
    - Resize
    - Convert to grayscale
    - Normalize
    """
    # Open image using Pillow
    img = Image.open(image_path)
    
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Resize image
    img_resized = img_gray.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img_resized) / 255.0
    
    # Expand dimensions to match model input shape (batch_size, width, height, channels)
    # Expected shape: (1, 128, 128, 1)
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def check_general_object(image_path):
    """
    Uses MobileNetV2 to predict the general object in the image.
    Returns the top label and its confidence.
    """
    if general_model is None:
        return None, 0.0

    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = mobilenet_preprocess(img_array)

        preds = general_model.predict(img_array)
        decoded = decode_predictions(preds, top=1)[0][0]
        # decoded is a tuple (class_id, class_name, prob)
        return decoded[1], float(decoded[2])
    except Exception as e:
        print(f"Error checking general object: {e}")
        return None, 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Server configuration error.'}), 500

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['file']
    
    # If the user does not select a file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # First, check if the image is actually currency using MobileNetV2
            general_label, general_conf = check_general_object(filepath)
            
            # Strict list of valid ImageNet classes that might represent currency
            currency_keywords = ['paper_money', 'wallet', 'envelope', 'cash_machine', 'safe', 'purse', 'obelisk', 'library', 'maze', 'menu']
            
            # If the top prediction is not one of these currency/money correlates, 
            # we only reject it if we are somewhat confident (> 12%). This allows flat scans 
            # or collages (which confuse MobileNet) to still pass through to your custom CNN.
            if general_label is not None and general_label not in currency_keywords:
                if general_conf > 0.12:
                    friendly_label = general_label.replace('_', ' ').title()
                    return jsonify({
                        'success': True,
                        'result': f"Not Currency ({friendly_label})",
                        'confidence': f"{(general_conf * 100):.2f}%",
                        'is_real': False, # Treat as not valid
                        'filename': filename,
                        'image_url': f"/static/uploads/{filename}"
                    })
                # If general_conf <= 0.12, we let it pass. It's likely a complex currency layout.
            
            # Preprocess the image
            processed_img = preprocess_image(filepath)
            
            # Predict
            # Model returns probability between 0 and 1
            prediction = model.predict(processed_img)
            probability = float(prediction[0][0])
            
            # Since alphabetical class mode puts 'fake' as 0 and 'real' as 1
            is_real = probability > 0.5
            
            # Calculate confidence
            if is_real:
                confidence = probability * 100
                result_text = "Real Currency"
            else:
                confidence = (1 - probability) * 100
                result_text = "Fake Currency"
                
            # Formatting the response
            return jsonify({
                'success': True,
                'result': result_text,
                'confidence': f"{confidence:.2f}%",
                'is_real': bool(is_real),
                'filename': filename,
                'image_url': f"/static/uploads/{filename}"
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

if __name__ == '__main__':
    # Get port from environment variable for cloud deployment (e.g., Render, Heroku)
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)
