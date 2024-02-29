import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


class HandDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("model/keras_model.h5", "model/labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_output = img.copy()
        hands, _ = self.detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            img_white = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255

            img_crop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
            img_crop_shape = img_crop.shape

            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = self.imgSize / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_cal, self.imgSize))
                img_resize_shape = img_resize.shape
                w_gap = math.ceil((self.imgSize - w_cal) / 2)
                img_white[:, w_gap: w_cal + w_gap] = img_resize
                prediction, index = self.classifier.getPrediction(img_white, draw=False)
                st.write(prediction, index)

            else:
                k = self.imgSize / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (self.imgSize, h_cal))
                img_resize_shape = img_resize.shape
                h_gap = math.ceil((self.imgSize - h_cal) / 2)
                img_white[h_gap: h_cal + h_gap, :] = img_resize
                prediction, index = self.classifier.getPrediction(img_white, draw=False)

            cv2.rectangle(img_output, (x - self.offset, y - self.offset - 70),
                          (x - self.offset + 400, y - self.offset + 60 - 50), (0, 255, 0), cv2.FILLED)

            cv2.putText(img_output, self.labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(img_output, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
                          (0, 255, 0), 4)

        return img_output


def main():
    st.title("Indian Sign Language Recognition")

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=HandDetectionTransformer)

    if webrtc_ctx.video_transformer:
        st.video(webrtc_ctx.video_transformer)


if __name__ == "__main__":
    main()
