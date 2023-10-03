from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
MODEL_FOLDER = 'static/models/'

# Load models
unet_model = load_model(os.path.join(MODEL_FOLDER, 'unet_best_model.h5'))
efficientnet_model = load_model(os.path.join(MODEL_FOLDER, 'efficientnet_best_model.h5'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        model_type = request.form.get("model_choice")
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            output_filename, segmentation_percentage, interpretation = predict(filepath, model_type)
            return render_template('results.html', uploaded_img=filename, mask_img=output_filename, segmentation_percentage = round(segmentation_percentage, 2), interpretation=interpretation)
    return render_template('index.html')


def preprocess_unet(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_efficientnet(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def post_process_mask(pred_mask):
    mask = np.squeeze(pred_mask) * 255
    mask = mask.astype(np.uint8)
    return mask


def compute_segmentation_percentage(mask_array):
    """Compute the percentage of the skin area segmented."""
    # Count the number of pixels in the segmentation mask
    segmented_pixels = np.sum(mask_array > 0.5)

    # Total number of pixels in the image
    total_pixels = mask_array.size

    # Compute the percentage
    percentage = (segmented_pixels / total_pixels) * 100
    return percentage

def categorize_segmentation(percentage):
    """Provide a textual interpretation based on the segmentation percentage."""
    if percentage < 10:
        return ("Minimal Segmentation. The segmented regions appear to be benign. Monitor any changes and consult if needed.")
    elif percentage < 50:
        return ("Moderate Segmentation. The segmented regions are potentially atypical. It's advisable to consult a dermatologist.")
    else:
        return ("Significant Segmentation. Significant areas of concern detected. Immediate consultation with a medical professional is recommended.")

def predict(img_path, model_type):
    if model_type == "unet":
        input_array = preprocess_unet(img_path)
        preds = unet_model.predict(input_array)
    elif model_type == "efficientnet":
        input_array = preprocess_efficientnet(img_path)
        preds = efficientnet_model.predict(input_array)
    else:
        raise ValueError("Unexpected model_type value: " + str(model_type))

    mask = post_process_mask(preds)
    output_filename = "mask_" + os.path.basename(img_path)
    output_filepath = os.path.join(RESULT_FOLDER, output_filename)
    mask_img = Image.fromarray(mask)
    mask_img.save(output_filepath)

    segmentation_percentage = compute_segmentation_percentage(mask)
    interpretation = categorize_segmentation(segmentation_percentage)


    return output_filename, segmentation_percentage, interpretation

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
