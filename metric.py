import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

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

def preprocess_image(image_path):
    """
    Load and preprocess an image to the required grayscale format.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        image (numpy array): Preprocessed grayscale image of size (48, 48, 1).
    """
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((48, 48))  # Resize to 48x48
    image_array = np.array(image, dtype='float32')  # Convert to numpy array
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension (48, 48, 1)
    return image_array

def predict_emotion(model, image):
    """
    Predict the emotion for a given image using the loaded model.
    
    Args:
        model: Trained Keras model.
        image (numpy array): Preprocessed grayscale image of size (48, 48, 1).
    
    Returns:
        prediction (str): Predicted emotion label.
    """
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    predictions = model.predict(image)
    predicted_emotion = np.argmax(predictions)
    return predicted_emotion, predictions

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and display a confusion matrix.
    
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of emotion class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(y_true, y_pred, class_names):
    """
    Display the classification report with precision, recall, and F1-score for each class.
    
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of emotion class names.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n")
    print(report)

def visualize_correct_and_misclassified(X_test, y_true, y_pred, class_names, num_images=10):
    """
    Display a grid of correctly and misclassified images.
    
    Args:
        X_test (numpy array): Test images.
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of emotion class names.
        num_images (int): Number of correct and misclassified images to display.
    """
    correct = np.where(np.array(y_true) == np.array(y_pred))[0]
    misclassified = np.where(np.array(y_true) != np.array(y_pred))[0]
    
    print(f"Number of Correctly Classified Images: {len(correct)}")
    print(f"Number of Misclassified Images: {len(misclassified)}")
    
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(correct[:num_images]):
        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap='gray')
        plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", color="green")
        plt.axis('off')
    plt.suptitle('Correctly Classified Images', fontsize=16)
    plt.show()
    
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(misclassified[:num_images]):
        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap='gray')
        plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", color="red")
        plt.axis('off')
    plt.suptitle('Misclassified Images', fontsize=16)
    plt.show()

import os
import numpy as np

# Preprocess images
def load_images_from_directory(directory_path, class_names):
    """
    Load images from the specified directory and preprocess them.
    
    Args:
        directory_path (str): Path to the directory containing class subfolders.
        class_names (list): List of emotion class names.

    Returns:
        images (numpy array): Preprocessed images as numpy arrays.
        labels (list): Corresponding labels for the images.
    """
    images = []
    labels = []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(directory_path, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            try:
                image = preprocess_image(img_path)  # Use the preprocess_image function
                images.append(image)
                labels.append(class_idx)  # Assign label based on folder name
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    return np.array(images), labels


if __name__ == "__main__":
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    model = load_emotion_model('emotion_cnn_model_v2.h5')
    
    # Example usage of the new preprocess_image function
    # image = preprocess_image('path/to/your/image.jpg')
    # emotion_label, predictions = predict_emotion(model, image)
    # print("Predicted Emotion:", class_names[emotion_label])
    
    print("Model loader and visualization script ready.")

    # Load train and test datasets
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    train_dir = 'archive/train'
    test_dir = 'archive/test'

    X_train, y_train = load_images_from_directory(train_dir, class_names)
    X_test, y_true = load_images_from_directory(test_dir, class_names)

    # Generate predictions using the loaded model (suppress verbose output)
    y_pred = [np.argmax(model.predict(np.expand_dims(image, axis=0) / 255.0, verbose=0)) for image in X_test]

    # Plot and visualize results
    plot_classification_report(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)
    visualize_correct_and_misclassified(X_test, y_true, y_pred, class_names)

