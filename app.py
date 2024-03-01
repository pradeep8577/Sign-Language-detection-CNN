import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import mediapipe as mp
import streamlit as st


model = load_model('SignLanguage.h5',compile=False)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

# List of class names (update with your classes)
classes = ['A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X','Y','Z','del','nothing','space']


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 150)  # Coordinates (x, y) of the top-left corner of the text
font_scale = 1
color = (255, 255, 255)  # White color
thickness = 2
line_type = cv2.LINE_AA

st.title("Sign Language Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])


word = ''
# Initialize the camera capture
cap = cv2.VideoCapture(0)


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

while run:
     ret, frame = cap.read()
     if not ret:
      break

     # FRAME_WINDOW.image(frame)

     frame_height, frame_width, _ = frame.shape

     # Convert frame to RGB for Mediapipe
     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


     # Perform hand detection with Mediapipe
     results = hands.process(frame_rgb)
     hand_boxes = []
     if results.multi_hand_landmarks:
      hand_boxes = []

      for landmarks in results.multi_hand_landmarks:
       # Calculate bounding box coordinates
       x_min = int(min(landmark.x for landmark in landmarks.landmark) * frame_width)
       x_max = int(max(landmark.x for landmark in landmarks.landmark) * frame_width)
       y_min = int(min(landmark.y for landmark in landmarks.landmark) * frame_height)
       y_max = int(max(landmark.y for landmark in landmarks.landmark) * frame_height)

       width = x_max - x_min
       height = y_max - y_min

       hand_boxes.append([x_min, y_min, width, height])

      # Perform similar operations as before:
      # - Extract larger hand region
      # - Make predictions using your model
      # - Annotate the frame
     currentclass = ''
     for box in hand_boxes:
         x, y, w, h = box
         center_x = x + w // 2
         center_y = y + h // 2

         # Define the size of the larger hand region
         larger_size = 250

         # Calculate the new coordinates for the larger hand region
         new_x = max(center_x - larger_size // 2, 0)
         new_y = max(center_y - larger_size // 2, 0)
         # Extract hand region
         hand_region = frame[new_y:new_y + larger_size, new_x:new_x + larger_size]
         im = Image.fromarray(hand_region, 'RGB')

         # Resize the image to match your model's input size
         im = im.resize((200, 200))
         img_array = np.array(im)
         img_array = img_array / 255.0
         # Expand dimensions to match the model's input shape
         img_array = np.expand_dims(img_array, axis=0)

         # Make predictions using the loaded model
         predictions = model.predict(img_array)
         prediction = np.argmax(predictions[0])
         prop = predictions[0][prediction]
         predicted_class = classes[prediction]
         currentclass = predicted_class

         cv2.rectangle(frame, (new_x, new_y), (new_x + larger_size, new_y + larger_size), (0, 255, 0), 2)
         if (prop > 0.980):
            cv2.putText(frame, str(predicted_class) + str(prop), org, font, font_scale, color, thickness, line_type)
     # Display the frame
     # cv2.imshow("Hand Detection and Sign Classification", frame)

     # Display the processed frame in Streamlit
     FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)

     # Check for key press ('q' to exit)
     key = cv2.waitKey(1)
     if key == ord('a'):
      print(currentclass)
      word = word + currentclass

     if key == ord('q'):
      print(word)
      break

# Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()