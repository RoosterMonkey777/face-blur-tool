from flask import Flask, request, redirect, url_for, render_template
import os
import cv2
from mtcnn import MTCNN
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust logging level as needed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    process_images(app.config['UPLOAD_FOLDER'])
    return redirect(url_for('index'))

def process_images(folder):
    detector = MTCNN()
    
    for filename in os.listdir(folder):
        if allowed_file(filename):
            filepath = os.path.join(folder, filename)
            logging.info(f"Processing image: {filename}")  # Log the filename
            image = cv2.imread(filepath)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb_image)

            for detection in detections:
                x, y, width, height = detection['box']
                
                # Adjust bounding box to encompass head and ears
                x -= int(width * 0.1)  # Expand left side of the bounding box
                y -= int(height * 0.2)  # Expand top side of the bounding box
                width += int(width * 0.2)  # Expand right side of the bounding box
                height += int(height * 0.3)  # Expand bottom side of the bounding box
                
                # Ensure coordinates are within image boundaries
                x, y = max(x, 0), max(y, 0)
                x2, y2 = min(x + width, image.shape[1]), min(y + height, image.shape[0])
                face_image = image[y:y2, x:x2]
                
                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(face_image, (199, 199), 50)
                image[y:y2, x:x2] = blurred_face
            
            cv2.imwrite(filepath, image)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')