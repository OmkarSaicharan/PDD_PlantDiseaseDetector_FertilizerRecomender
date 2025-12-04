from flask import Flask, render_template, request, redirect, send_from_directory, url_for, render_template_string
import numpy as np
import json
import uuid
import tensorflow as tf
import os
from contour import find_leaf_contours

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# Label list
label = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Single diseased plant leaf allowed','Blueberry___healthy','Cherry___Powdery_mildew',
    'Cherry___healthy','Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust',
    'Corn___Northern_Leaf_Blight','Corn___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy',
    'Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# Load JSON mapping
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# Serve uploaded images
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

# Home route
@app.route('/', methods=['GET'])

def home():
    return render_template("home.html")

# Extract features
def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

# Predict disease
def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

# def model_predict(image):
#     img = extract_features(image)
#     prediction = model.predict(img)
#     pred_label = label[prediction.argmax()]
#     return {
#         "name": pred_label,
#         "cause": plant_disease[pred_label]["cause"],
#         "cure": plant_disease[pred_label]["cure"]
#     }

# Upload route (GET+POST)
@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']

        # Save uploaded file
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}_{image.filename}"
        image.save(temp_name)

        # Prediction
        prediction = model_predict(temp_name)

        # Contour output always saved with same name
        contour_output = "contour_output.jpg"
        find_leaf_contours(temp_name, contour_output)

        return render_template(
            'home.html',
            result=True,
            imagepath="/" + temp_name,
            prediction=prediction,
            contourimg="/contour_output.jpg"
        )
    else:
        return redirect('/')

# Serve any file (for contour_output.jpg etc.)
@app.route('/<path:filename>')
def serve_any(filename):
    return send_from_directory('.', filename)

# ---------------- FEATURE PAGES ---------------- #

@app.route('/fast-disease-detection')
def fast():
    return render_template("fast_disease_detection.html")

@app.route('/dashboard-history')
def dashboard():
    return render_template("dashboard_history.html")

@app.route('/treatment-guidance')
def treatment():
    return render_template("treatment_guidance.html")

@app.route('/user-profile-settings')
def profile():
    return render_template("user_profile_settings.html")

@app.route('/support-contact')
def support():
    return render_template("support_contact.html")

@app.route('/about')
def about():
    return render_template("about.html")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
