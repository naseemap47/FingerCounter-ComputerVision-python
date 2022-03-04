import cv2
import mediapipe as mp
import time
import os

cap = cv2.VideoCapture(0)
mp_hand = mp.solutions.hands
hand = mp_hand.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

p_time =0
tipe_id = [4, 8, 12, 16, 20]

folder_path = 'images'
img_list = os.listdir(folder_path)
# print(img_list)

overlay_list = []
for img_path in img_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(image)
print(len(overlay_list))

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)
    # print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        hand_landmarks_list = []
        for hand_lm in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_lm.landmark):
                height, width, channel = img.shape
                x, y = int(lm.x * width), int(lm.y * height)
                hand_landmarks_list.append([id, x, y])
                # print(hand_landmarks_list)
                if len(hand_landmarks_list) > 20:
                    finger = []
                    # Thumb
                    if hand_landmarks_list[tipe_id[0]][1] < hand_landmarks_list[tipe_id[0] - 1][1]:
                        finger.append(1)
                    else:
                        finger.append(0)

                    # 4 fingers
                    for id in range(1, 5):
                        if hand_landmarks_list[tipe_id[id]][2] < hand_landmarks_list[tipe_id[id]-2][2]:
                            finger.append(1)
                        else:
                            finger.append(0)
                    # print(finger)
                    total_fingers = finger.count(1)
                    # print(total_fingers)
                    cv2.rectangle(
                        img, (20, 225), (170, 425),
                        (0, 255, 0), cv2.FILLED,
                    )
                    cv2.putText(
                        img, str(total_fingers),
                        (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 15
                    )
            mp_draw.draw_landmarks(img, hand_lm, mp_hand.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(
        img, f'FPS: {int(fps)}',
        (10, 60), cv2.FONT_HERSHEY_PLAIN,
        3,(255, 0, 255), 3
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)