import cv2 as cv
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as drawing

# Initialize the Hands module
hands = mpHands.Hands(
    static_image_mode=False,  # Corrected typo here
    max_num_hands=2,          # Maximum number of hands to detect
    min_detection_confidence=0.5  # Confidence threshold for detection
)
#Get Hand Landmarks
def getHandlandMarks(img, draw):
    lmlist = [ ]
    hands= mpHands.Hands(
        static_image_mode=False, 
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    frameRgb =cv.cvtColor(img, cv.COLOR_BGR2RGB)
    handsDetected= hands.process (frameRgb)
    if handsDetected.multi_hand_landmarks: 
        for lanmarks in handsDetected.multi_hand_landmarks:
            #print(lanmarks)
            for id, lm in enumerate (lanmarks.landmark):
                #print(id, lm)
                h,w,c= img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append((id, cx, cy))
                #print(lmlist)
        if draw :
            drawing.draw_landmarks(
                img,
                lanmarks,
                mpHands.HAND_CONNECTIONS

            )
                
    return lmlist


def fingerCount(lmlist):
    count = 0
    if lmlist[8][2] < lmlist[6][2]:
        count +=1
    if lmlist[12][2] < lmlist[10][2]:
        count +=1
    if lmlist[16][2] < lmlist[14][2]:
        count +=1
    if lmlist[20][2] < lmlist[18][2]:
        count +=1
    if lmlist[4][1] < lmlist[2][1]:
        count +=1

    return count



# Start the webcam
cam = cv.VideoCapture(0)

while True:
    success, frame = cam.read()
    if not success:
        print("Camera not detected..!")
        continue
    lmlist = getHandlandMarks (img=frame, draw=False)
    if lmlist:
        #print(lmlist)
        fc= fingerCount(lmlist=lmlist)
        #print(fc)
        cv.rectangle(frame, (400,10), (600,250), (0,0,0), -1)
        cv.putText(frame, str(fc), (400,250), cv. FONT_HERSHEY_PLAIN, 20, (0,255,255), 30)





    # # Flip the frame horizontally (for mirror effect)
    # frame = cv.flip(frame, 1)
    
    # # Convert the frame to RGB (as MediaPipe requires RGB input)
    # frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # # Process the frame and detect hands
    # handsDetected = hands.process(frameRGB)
    
    # # If hands are detected, draw landmarks
    # if handsDetected.multi_hand_landmarks:
    #     for hand_landmarks in handsDetected.multi_hand_landmarks:
    #         # Print landmark details (optional)
    #         for idx, landmark in enumerate(hand_landmarks.landmark):
    #             print(f"Landmark {idx}: x = {landmark.x}, y = {landmark.y}, z = {landmark.z}")

    #         # Draw landmarks and hand connections
    #         drawing.draw_landmarks(
    #             image=frame,
    #             landmark_list=hand_landmarks,
    #             connections=mpHands.HAND_CONNECTIONS  # Draw hand skeleton
    #         )
    

    cv.imshow("fingercounting", frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
