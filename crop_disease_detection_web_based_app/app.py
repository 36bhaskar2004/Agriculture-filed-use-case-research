import os
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
MODEL_PATH = 'plant_disease_model.pth'
WEATHER_API_KEY = '65fd251c23a2bd94ffa55a03b07cfe6f'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = None, []

# ------------------- Load Model -------------------
def load_model():
    global model, class_names
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_names = checkpoint['class_names']
    return model.to(device), class_names

model, class_names = load_model()

# ------------------- Disease Info -------------------
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "description": "Fungal disease causing dark, scaly lesions on leaves and fruit",
        "recommendations": [
            "Apply fungicides containing myclobutanil",
            "Remove infected leaves and fruit",
            "Plant resistant varieties"
        ],
        "fertilizers": ["NPK 10-10-10", "Calcium-rich fertilizer"],
        "severity": "High"
    },
    "Tomato___Early_blight": {
        "description": "Fungal disease causing concentric rings on leaves",
        "recommendations": [
            "Rotate crops every 2-3 years",
            "Use mulch to prevent soil splash",
            "Apply copper fungicides"
        ],
        "fertilizers": ["NPK 5-5-5", "High-potassium fertilizer"],
        "severity": "Moderate"
    },
}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------- Prediction Functions -------------------
def predict_disease(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top3_probs, top3_idxs = torch.topk(probs, 3)

            predictions = []
            for i in range(3):
                class_name = class_names[top3_idxs[i]]
                confidence = float(top3_probs[i])
                info = DISEASE_INFO.get(class_name, {
                    "description": "General plant care needed",
                    "recommendations": ["Monitor plant health regularly"],
                    "fertilizers": ["Balanced NPK fertilizer"],
                    "severity": "Low"
                })
                predictions.append({
                    "name": class_name.replace('___', ' ').replace('_', ' '),
                    "confidence": confidence,
                    "info": info
                })
        return predictions
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None
    
#live prediction
def predict_frame(frame):
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_idx = torch.argmax(probs).item()
            class_name = class_names[top_idx]
            confidence = float(probs[top_idx])
        return f"{class_name} ({confidence:.2f})"
    except Exception as e:
        print("Frame prediction error:", str(e))
        return "Error"

# ------------------- Weather -------------------
def get_weather(city):
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&appid={WEATHER_API_KEY}"
        geo_res = requests.get(geo_url).json()
        if not geo_res:
            return None
        lat, lon = geo_res[0]['lat'], geo_res[0]['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        return requests.get(weather_url).json()
    except:
        return None
# ------------------- Live Camera Streaming -------------------
def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run prediction
            prediction = predict_frame(frame)

            # Overlay prediction text on the frame
            cv2.putText(frame, prediction, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------- Routes -------------------
@app.route('/')
def home():
    return render_template('index.html')
# live
@app.route('/live')
def live():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predictions = predict_disease(filepath)
        if not predictions:
            return jsonify({'error': 'Prediction failed'}), 500
        return jsonify({'predictions': predictions, 'image_url': filepath})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_video', methods=['POST'])
def predict_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        if file and file.filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return jsonify({'error': 'Cannot open video file'}), 500

            frame_rate = 5
            count, predictions = 0, []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count % frame_rate == 0:
                    frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    preds = predict_disease(frame_path)
                    if preds:
                        predictions.append(preds[0])
                count += 1
            cap.release()

            if not predictions:
                return jsonify({'error': 'No predictions from video'}), 500

            # majority vote
            disease_counts = {}
            for pred in predictions:
                name = pred["name"]
                disease_counts[name] = disease_counts.get(name, 0) + 1
            final_disease = max(disease_counts, key=disease_counts.get)

            return jsonify({
                'final_prediction': final_disease,
                'frame_predictions': predictions,
                'video_url': filepath
            })
        return jsonify({'error': 'Invalid video type'}), 400
    except Exception as e:
        import traceback
        print("Video prediction error:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/weather', methods=['POST'])
def weather():
    city = request.form.get('city')
    if not city:
        return jsonify({'error': 'No city provided'}), 400
    weather_data = get_weather(city)
    if not weather_data:
        return jsonify({'error': 'Weather data unavailable'}), 500
    return jsonify({
        'city': weather_data.get('name'),
        'temp': weather_data['main']['temp'],
        'humidity': weather_data['main']['humidity'],
        'conditions': weather_data['weather'][0]['main'],
        'icon': weather_data['weather'][0]['icon']
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(debug=True)


    
