import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import random

LastChngeVeloTime = 0
# Initialize webcam
#1
webcam = cv.VideoCapture(0)
# Initialize hand tracking
#2
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

# Initialize paddle and puck positions
#3
paddle = [0, 0]
puck = [200, 200]

# Initial velocity
initial_puck_velocity = [1,1]
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
#4
target_image = cv.imread('target.png')
target_image = cv.resize(target_image, (30,30))

donut2gray = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(donut2gray, 1, 255, cv.THRESH_BINARY)
# Initialize 5 target positions randomly(remember assignment 2!!)
#5
target_positions = []
for i in range(5):
    a, b = random.randint(0,int(webcam.get(cv.CAP_PROP_FRAME_WIDTH))), random.randint(0,int(webcam.get(cv.CAP_PROP_FRAME_HEIGHT)))
    target_positions.append([a,b])
print(target_positions)
left_targets = 5
# Initialize score
#6
score = 0

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds

# Function to check if the puck is within a 5% acceptance region of a target
def is_within_acceptance(puck, target, acceptance_percent=5):
  #complete the function
  #7
    if target[0]-(acceptance_percent/100)*target[0]<puck[0]<target[0]+(acceptance_percent/100)*target[0] and target[1]-(acceptance_percent/100)*target[1]<puck[1]<target[1]+(acceptance_percent/100)*target[1]:
        return 1

while True:
    # Calculate remaining time and elapsed time in minutes and seconds   
    #9
    current_time = time.time()
    remaining_time = 30-(current_time-start_time)
    elapsed_time = 30-remaining_time

    # Read a frame from the webcam
    #10
    isTrue, frame = webcam.read()
    # Flip the frame horizontally for a later selfie-view display
    #11
    frame = cv.flip(frame,1)
    # Convert the BGR image to RGB
    #12
    frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Process the frame with mediapipe hands
    #13
    hands_results = hands.process(frame_RGB)



    # Update paddle position based on index finger tip
    #14
    if hands_results.multi_hand_landmarks:
        for handLms in hands_results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if id == 8:
                    paddle[0] = int(lm.x*webcam.get(cv.CAP_PROP_FRAME_WIDTH))
                    paddle[1] = int(lm.y*webcam.get(cv.CAP_PROP_FRAME_HEIGHT))
                    frame = cv.rectangle(frame, (paddle[0]-50,paddle[1]-5),(paddle[0]+50,paddle[1]+5),(0,225,0),-1)
    else :
        paddle[0] = 0
        paddle[1] = 0
    # Update puck position based on its velocity
    #15
    puck[0] += puck_velocity[0]*elapsed_time
    puck[1] += puck_velocity[1]*elapsed_time
    # Check for collisions with the walls
    #16
    if puck[0]>webcam.get(cv.CAP_PROP_FRAME_WIDTH) or puck[0]<0 :
        puck_velocity[0] = -puck_velocity[0]
    if puck[1]>webcam.get(cv.CAP_PROP_FRAME_HEIGHT) or puck[1]<0 :
        puck_velocity[1] = -puck_velocity[1]
    # print(puck_velocity)
    # Check for collisions with the paddle
    #17
    if (time.time() - LastChngeVeloTime)>1 and hands_results.multi_hand_landmarks:
        if paddle[1]-1<puck[1]<puck[1]+1 and paddle[0]-50<puck[0]<puck[0]+50:
            print(paddle,puck)
            puck_velocity[1] = -puck_velocity[1]
            LastChngeVeloTime = time.time()

    # Check for collisions with the targets(use is_within_acceptance)    
    #18
            # Increase the player's score
            # Remove the hit target
            # Increase puck velocity after each hit by 2(you will have to increase both x and y velocities

    for i in range(left_targets):
        if is_within_acceptance(puck,target_positions[left_targets-1-i]):
            score += 1
            target_positions.pop(left_targets-1-i)
            left_targets -= 1

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    #19
    frame = cv.circle(frame,(int(puck[0]),int(puck[1])),15,(0,0,225),-1)
    for i in range(left_targets):
        frame[target_positions[i][0]-15:target_positions[i][0]+15, target_positions[i][1]-15:target_positions[i][1]+15] = target_image[:,:]
    # FOR REFERENCE:
   #  for target_position in target_positions:
   #      target_roi = frame[target_position[1]:target_position[1] + target_image.shape[0],
   #                        target_position[0]:target_position[0] + target_image.shape[1]]
   #      alpha = target_image[:, :] / 255.0
   #      beta = 1.0 - alpha
   #      for c in range(0, 3):
   #          target_roi[:, :, c] = (alpha * target_image[:, :, c] +
   #                                beta * target_roi[:, :, c])

    # Display the player's score on the frame
    #20
    frame = cv.putText(frame, f"Score = {score}", (10,70),cv.FONT_HERSHEY_TRIPLEX,1.0,(225,0,0),2)

    # Display the remaining time on the frame
    #21
    frame = cv.putText(frame,f"Time Left:{int(remaining_time)}",(10,90),cv.FONT_HERSHEY_TRIPLEX,1.0,(225,0,0),2)
    # Check if all targets are hit or time is up
    #22

    # Display the resulting frame
    #23
    cv.imshow("webcam", frame)

    # Exit the game when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
#24
webcam.release()
cv.destroyAllWindows()
cv.waitKey(0)

