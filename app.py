from flask import Flask, flash, request, render_template, redirect
import PIL
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = 'C:/Users/farouk/Desktop/Skin-Cancer-Classification-main/Skin-Cancer-Classification-main/best_model1.h5'
model = load_model(model_path)

dic = {0: 'benign', 1: 'malignant'}

@app.route('/', methods=["GET", "POST"])
def runhome():
    return render_template('home.html')

@app.route('/showresult', methods=["POST"])
def show():
    if 'pic' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    pic = request.files['pic']
    if pic.filename == '':
        flash('No selected file')
        return redirect(request.url)

    try:
        # Process the uploaded image
        inputimg = PIL.Image.open(pic)
        inputimg = inputimg.resize((28, 28))  # Resize to match model input size
        img = np.array(inputimg) / 255.0  # Normalize pixel values (assuming model expects inputs in range [0, 1])
        img = img.reshape(-1, 28, 28, 3)  # Reshape to match model input shape

        # Get model prediction
        result = model.predict(img)
        result = result.tolist()

        max_prob = max(result[0])
        class_ind = result[0].index(max_prob)
        predicted_class = dic[class_ind]

        # Define information based on predicted class
        info = ""
        if predicted_class == 'benign':
            info = "Actinic keratosis also known as solar keratosis or senile keratosis are names given to intraepithelial keratinocyte dysplasia. As such they are a pre-malignant lesion or in situ squamous cell carcinomas and thus a malignant lesion."
        elif predicted_class == 'malignant':
            info = "Basal cell carcinoma is a type of skin cancer..."

        return render_template('results.html', result=predicted_class, info=info)

    except Exception as e:
        flash(f"Error processing image: {e}")
        return redirect(request.url)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
