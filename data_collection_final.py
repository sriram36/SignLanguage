
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Use local dataset directory
dataset_base = "AtoZ_3.1"
c_dir = 'A'
count = len(oss.listdir(oss.path.join(dataset_base, c_dir)))

offset = 15
step = 1
flag = False
suv = 0

white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("white.jpg", white)


while True:
    try:

        ret, frame = capture.read()
        if not ret or frame is None:
            print("Failed to grab frame from camera.")
            continue
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        white = cv2.imread("white.jpg")


        if hands:
            hand = hands[0]
            print('DEBUG hand:', hand)  # Debug print to inspect structure
            if hand:  # Only process if hand is not empty
                if isinstance(hand, dict) and 'bbox' in hand:
                    x, y, w, h = hand['bbox']
                elif isinstance(hand, (list, tuple)) and len(hand) > 0 and isinstance(hand[0], dict) and 'bbox' in hand[0]:
                    x, y, w, h = hand[0]['bbox']
                else:
                    print(f"Skipping frame, unexpected hand structure: {hand}")
                    continue
                image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])
            else:
                # No hand detected in this frame
                continue

            handz,imz = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                # x1,y1,w1,h1=hand['bbox']
                os=((400-w)//2)-15
                os1=((400-h)//2)-15
                for t in range(0,4,1):
                    cv2.line(white,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(5,8,1):
                    cv2.line(white,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(9,12,1):
                    cv2.line(white,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(13,16,1):
                    cv2.line(white,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(17,20,1):
                    cv2.line(white,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                cv2.line(white, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[5][0]+os, pts[5][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)

                skeleton0=np.array(white)
                zz=np.array(white)
                for i in range(21):
                    cv2.circle(white,(pts[i][0]+os,pts[i][1]+os1),2,(0 , 0 , 255),1)

                skeleton1=np.array(white)

                cv2.imshow("1",skeleton1)

        frame = cv2.putText(frame, "dir=" + str(c_dir) + "  count=" + str(count), (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break


        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir)+1)
            if ord(c_dir)==ord('Z')+1:
                c_dir='A'
            flag = False
            count = len(oss.listdir(oss.path.join(dataset_base, c_dir)))

        if interrupt & 0xFF == ord('a'):
            if flag:
                flag=False
            else:
                suv=0
                flag=True

        print("=====",flag)
        if flag==True:

            if suv==180:
                flag=False
            if step%3==0:

                save_path = oss.path.join(dataset_base, c_dir, f"{count}.jpg")
                cv2.imwrite(save_path, skeleton1)

                count += 1
                suv += 1
            step+=1



    except Exception:
        print("==",traceback.format_exc() )

capture.release()
cv2.destroyAllWindows()