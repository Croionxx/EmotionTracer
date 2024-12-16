import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image processing

# Load the pre-trained model
model_path = 'emotion_cnn_model_v2.h5'  # Update this if your model file has a different name
model = tf.keras.models.load_model(model_path)

# Emotion labels corresponding to the class indices
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(image_path):
    """
    Preprocesses the image to match the model's expected input.
    
    Args:
        image_path (str): Path to the image to be processed.
    
    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    # Load the image in grayscale mode (since the model was trained on grayscale images)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Resize the image to 48x48 (since the model was trained on 48x48 images)
    image = cv2.resize(image, (48, 48))
    
    # Normalize pixel values to [0, 1] range
    image = image / 255.0
    
    # Reshape the image to (1, 48, 48, 1) to match the input shape of the model
    image = np.reshape(image, (1, 48, 48, 1))
    
    return image

def predict_emotion(image_path):
    """
    Predicts the emotion from the input image using the pre-trained model.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: The predicted emotion label.
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Make a prediction using the model
        predictions = model.predict(processed_image)
        
        # Get the index of the emotion with the highest probability
        predicted_index = np.argmax(predictions)
        
        # Get the corresponding emotion label
        predicted_emotion = emotion_labels[predicted_index]
        
        print(f"Predicted Emotion: {predicted_emotion}")
        return predicted_emotion
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Prompt the user to input the path to the image file
    image_path = input("Enter the path to the image file: ")
    
    # Predict the emotion from the image
    predict_emotion(image_path)
