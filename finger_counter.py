import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=1)

p_time =0

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)
    print(result.multi_hand_landmarks)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(
        img, f'FPS: {int(fps)}',
        (10, 60), cv2.FONT_HERSHEY_PLAIN,
        3,(255, 0, 0), 3
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)