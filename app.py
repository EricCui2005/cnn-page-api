from flask import Flask, jsonify, request
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io


# Initialize Flask app
app = Flask(__name__)

# Sample route
@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Flask API',
        'status': 'success'
    })

# Returns a model summary
@app.route('/model-summary')
def model_summary():
    try:
        # Load the model
        model = keras.models.load_model('model.h5')
        
        # Get model summary as string
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary_str = '\n'.join(summary_list)
        
        return jsonify({
            'status': 'success',
            'model_summary': summary_str
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Returns prediction probabilities for an image
@app.route('/classify', methods=['POST'])
def classify():
    
    class_mappings = {
        0: "tench",
        1: "English springer",
        2: "cassette player",
        3: "chain saw",
        4: "church",
        5: "French horn",
        6: "garbage truck",
        7: "gas pump",
        8: "golf ball",
        9: "parachute"
    }
    try:
        # Get the image data from the request
        image_data = request.get_data()
        
        # Convert binary data to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Resize image to match model's expected input size
        # Note: You may need to adjust these dimensions based on your model
        image = image.resize((64, 64))
        
        # Convert to numpy array and preprocess
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Load model and make prediction
        model = keras.models.load_model('model.h5')
        
        # Explicitly compile the model to suppress the warning
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logits = model.predict(img_array)
        
        # Convert logits to probabilities using softmax
        probabilities = tf.nn.softmax(logits).numpy()[0]
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        print(probabilities)
        
        return jsonify({
            'status': 'success',
            'predicted_class': class_mappings[int(predicted_class)],
            'confidence': confidence,
            'class_probabilities': probabilities.tolist()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)