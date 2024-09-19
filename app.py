from flask import Flask, request, jsonify, render_template
import os
import torch
from PIL import Image
import io

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def load_class_names(file_path='coco.names'):
    with open(file_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

class_names = load_class_names()

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

latest_detections = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global latest_detections
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        latest_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_image.jpg')
        
        file.save(file_path)
        
        file.seek(0)
        file.save(latest_file_path)
        
        try:
            with Image.open(latest_file_path) as img:
                img.verify()
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        results = model(file_path)
        
        detections = results.pandas().xyxy[0]
        detections['class_name'] = detections['class'].apply(lambda x: class_names[int(x)])
        
        latest_detections = detections.to_dict(orient='records')
        
        return jsonify(latest_detections)

@app.route('/search', methods=['POST'])
def search_object():
    global latest_detections
    
    request_data = request.get_json()
    search_term = request_data.get('search', '').lower()
    
    if not latest_detections:
        return jsonify({'error': 'No detections found'}), 404
    
    detected_classes = [d['class_name'].lower() for d in latest_detections]
    found = search_term in detected_classes
    
    return jsonify({'found': found})

if __name__ == '__main__':
    app.run(debug=True)
