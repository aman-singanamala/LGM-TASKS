import cv2
import streamlit as st

st.title("FILTERS")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    grey_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # invert image
    invert_cap = cv2.bitwise_not(grey_cap)
    #Blur Image
    blur_cap = cv2.GaussianBlur(invert_cap, (111,111), 0)
    #Invert Blur IMAGE
    invblur_cap = cv2.bitwise_not(blur_cap)
    sketch_cap = cv2.divide(grey_cap, invblur_cap, scale=256.0)
    FRAME_WINDOW.image(sketch_cap)
else:
    st.write('Stopped')