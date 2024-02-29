import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model/signlanguagedetectionmodel48x48.h5")  # Replace with your model path

# Define class labels
label_map = {0: 'A', 1: 'M', 2: 'N', 3: 'S', 4: 'T', 5: 'blank'}

# Function to extract image features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize video capture
cap = cv2.VideoCapture(0)

# Streamlit app
st.title("Real-Time Sign Language Detection")

if st.button("Start detection"):
    run = True
    while run:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Define prediction area and extract features
        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
        crop_frame = frame[40:300, 0:300]
        gray_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (48, 48))
        features = extract_features(resized_frame)

        # Make prediction and display results
        prediction = model.predict(features)
        predicted_label = label_map[prediction.argmax()]
        accuracy = "{:.2f}%".format(np.max(prediction) * 100)

        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        if predicted_label == 'blank':
            display_text = " "
        else:
            display_text = f"{predicted_label} {accuracy}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame in Streamlit
        st.image(frame, channels="BGR")

        # Check for keyboard input to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            run = False

    cap.release()
    cv2.destroyAllWindows()

st.text("Press 'Start detection' to begin")