import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the pre-trained model
@st.cache_resource
def load_emotion_model(model_path='emotion_cnn_model_v2.h5'):
    """
    Load the pre-trained CNN model from the specified path.
    
    Args:
        model_path (str): Path to the model file.
    
    Returns:
        model: Loaded Keras model.
    """
    model = load_model(model_path)
    print("Model loaded successfully from", model_path)
    return model

# Preprocess an image to grayscale format (48x48x1)
def preprocess_image(image):
    """
    Preprocess the uploaded image to grayscale and resize it to 48x48.
    
    Args:
        image (PIL Image): Input image uploaded by the user.
    
    Returns:
        image_array (numpy array): Preprocessed grayscale image of size (48, 48, 1).
    """
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((48, 48))  # Resize to 48x48
    image_array = np.array(image, dtype='float32')  # Convert to numpy array
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension (48, 48, 1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 48, 48, 1)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    return image_array

# Predict the emotion for the preprocessed image
def predict_emotion(model, image):
    """
    Predict the emotion for a given image using the loaded model.
    
    Args:
        model: Trained Keras model.
        image (numpy array): Preprocessed grayscale image of size (1, 48, 48, 1).
    
    Returns:
        prediction (str): Predicted emotion label.
    """
    predictions = model.predict(image)
    predicted_emotion = np.argmax(predictions)
    return predicted_emotion, predictions

# Streamlit app configuration
st.title("Emotion Recognition App")
st.write("Upload an image, and the model will predict the emotion.")

# Upload image via Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    with st.spinner("Processing and predicting the emotion..."):
        preprocessed_image = preprocess_image(image)
        model = load_emotion_model('emotion_cnn_model_v2.h5')
        predicted_emotion_index, predictions = predict_emotion(model, preprocessed_image)
        
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predicted_emotion = class_names[predicted_emotion_index]
        
        st.write("**Predicted Emotion:**", predicted_emotion)
        
        # Add emotion names to the bar chart
        prediction_data = {class_names[i]: predictions[0][i] for i in range(len(class_names))}
        st.bar_chart(prediction_data)
