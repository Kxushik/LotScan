import numpy as np # scientific computing
import matplotlib.pyplot as plt # plotting
import matplotlib.image as mpimg # reading images
from skimage.color import rgb2gray # converting rgb images to grayscale
import cv2
import math
import sys

if len(sys.argv)>2:
    print("Running Detection on Provided Images [Empty Parking Lot: "+sys.argv[1] +"] [Populated Parking Lot: "+sys.argv[2]+"]")
    path_empty_lot = sys.argv[1]
    path_cars_lot = sys.argv[2]
else:
    path_empty_lot = 'images/lot_empty.jpg'
    path_cars_lot = 'images/lot_cars.jpg'
    print("Running Detection on Default Images [Empty Parking Lot: "+path_empty_lot+"] [Populated Parking Lot: "+path_cars_lot+"]")


#Part 1 - Identifying Parking Spaces in Empty Parking Lot to Create Dictionary
##############################################################################
img1 = mpimg.imread(path_empty_lot)
img1_sliced = img1[:,:,:3]
img1 = img1.astype(np.uint8)

#Image Processing to Prepare For Canny Edge Detection
img1_gray = cv2.cvtColor(img1_sliced, cv2.COLOR_BGR2GRAY)
img1_blur = cv2.GaussianBlur(img1_gray, (5, 5), 1.505)

#Canny Edge Detection
img1_canny = cv2.Canny(img1_blur, 50, 150)

# Attempt to remove double lines with kernel and dilation and eroding
#Removing Double Lines using a Kernel, Dilation and Eroding Functions
kernel = np.ones((4,4),np.uint8)
img1_canny = cv2.dilate(img1_canny, kernel, iterations=2)
img1_canny = cv2.erode(img1_canny, kernel, iterations=3)

#Creating HoughLinesP to mark out the spaces
lines = cv2.HoughLinesP(image=img1_canny, rho=1, theta=np.pi/180, threshold=75, minLineLength=15, maxLineGap=90)
line_image = np.copy(img1)

#Modifying Co-ordinates to apply lines on the open ends of the parking spaces -> Spencer add comments
# Array to store north-side y-coordinates of vertical lines
heightsTop = []
# Array to store south-side y-coordinates of vertical lines
heightsBot = []

prevLine = lines[0]
for line in lines:
    for x1, y1, x2, y2 in line:
        # If current and previous lines are vertical, append their north side end-point to topArray, append their south side end-point to bottomArray
        if (x1 == x2 and prevLine[0][0] == prevLine[0][2]):
          heightsBot.append(y1)
          heightsTop.append(y2)
    prevLine = line

# Calculate median of heightsTop and heightsBottom
median = int(np.median(heightsTop))
median2 = int(np.median(heightsBot))

#List for all line coordinates
line_list = []

#Top and Bottom Lines for Parking Spaces (Open End)
cv2.line(line_image, (0, median), (600, median), (255, 0 , 0), 2)
cv2.line(line_image, (0, median2), (600, median2), (255, 0 , 0), 2)

#Append new lines to list
line_list.append([0, median, 600, median])
line_list.append([0, median2, 600, median2])

# Adjust y-value to be the median, if the lines are vertical (<4 slant)
for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        if (abs(lines[i][0][0] - lines[i][0][2]) < 4):
          lines[i][0][1] = median2
          lines[i][0][3] = median
        #Append the rest of the lines
        line_list.append([x1, lines[i][0][1], x2, lines[i][0][3]])
        cv2.line(line_image, (x1, lines[i][0][1]), (x2, lines[i][0][3]), (255, 0 , 0), 2)

#Draw lines onto image
for i in range(len(line_list)):
    cv2.line(line_image, (line_list[i][0], line_list[i][1]), (line_list[i][2], line_list[i][3]), (255, 0 , 0), 2)
    
#Transparent image kinda with lines
lines_edges = cv2.addWeighted(img1, 0.8, line_image, 1, 0)


#Processing Perpendicular Lines -> Kyle add comments
perpendicular_lines = []

#calculate angle between two lines (for perpendicularity)
def calculate_angle(line1, line2):
    #establishing vectors
    AB = (line1[2] - line1[0], line1[3] - line1[1])
    CD = (line2[2] - line2[0], line2[3] - line2[1])

    #dot product
    dot_product = AB[0] * CD[0] + AB[1] * CD[1]

    #magnitude of vectors
    magnitude_AB = np.sqrt(AB[0] ** 2 + AB[1] ** 2)
    magnitude_CD = np.sqrt(CD[0] ** 2 + CD[1] ** 2)

    #avoiding division by zero
    if magnitude_AB != 0 and magnitude_CD != 0:
        #cosine of the angle
        cos_angle = dot_product / (magnitude_AB * magnitude_CD)

        #converting angle to degrees
        angle_radians = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees
    return None



#finds which lines are perpendicular with each other
def find_perpendicular(lines):
    for i in range(len(lines)):
        for j in range(len(lines)):
            if (i != j):
                angle = calculate_angle(lines[i], lines[j])
                if (85 <= angle <= 95):
                    perpendicular_lines.append((lines[i], lines[j]))

find_perpendicular(line_list)
#print(perpendicular_lines)

#draw perpendicular lines on image
for i in range(len(perpendicular_lines)):
    x1, y1, x2, y2 = perpendicular_lines[i][0]
    x3, y3, x4, y4 = perpendicular_lines[i][1]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(line_image, (x3, y3), (x4, y4), (0, 255, 0), 2)

#Finding Intersection Points of Perpindicular Lines -> Kyle add comments
parking_spaces = {}

def find_intersections(perpendicular_lines):
    intersection_points = []
    #need to find every coordinate of the intersection on perpendicular lines and append it to a list of these coordinates
    for i in range(len(perpendicular_lines)):
        x1, y1, x2, y2 = perpendicular_lines[i][0] #line1
        x3, y3, x4, y4 = perpendicular_lines[i][1] #line2
        #check if line1 is vertical
        if x1 == x2:
            #if line2 is horizontal
            if y3 == y4:
                intersection_points.append((x1, y3))
            else:
                #calculate slope if line2 is not horizontal
                m2 = (y4 - y3) / (x4 - x3)
                b2 = y3 - m2 * x3
                y = m2 * x1 + b2
                intersection_points.append((x1, math.trunc(y)))
        #check if line1 is horizontal
        elif y1 == y2:
            #if line2 is vertical
            if x3 == x4:
                intersection_points.append((x3, y1))
            else:
                #calculate slope if line2 is not vertical
                m2 = (y4 - y3) / (x4 - x3)
                b2 = y3 - m2 * x3
                x = (y1 - b2) / m2
                intersection_points.append((math.trunc(x), y1))
    return intersection_points

intersection_points = find_intersections(perpendicular_lines)
#print(intersection_points)
#print(len(intersection_points))

#draw intersection points on image
for i in range(len(intersection_points)):
    cv2.circle(line_image, (int(intersection_points[i][0]), int(intersection_points[i][1])), 5, (0, 0, 255), -1)


# Kmeans Clustering to define 1 Coordinate Point per corner of parking space
#Convert intersection points to numpy array for opencv kmeans
intersection_points = np.array(intersection_points, dtype=np.float32)
#Criteria for kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#Kmeans clustering with 40 clusters
_, labels, centers = cv2.kmeans(intersection_points, 40, None, criteria, 10,  cv2.KMEANS_PP_CENTERS)

#convert centers to int32 for circles
centers = np.int32(centers)

for center in centers:
    cv2.circle(img1, tuple(center), 5, (255, 0, 0), -1)



#Define Parking Spaces with Clustered List

#Sort Y values first then X values
sortIndex = np.lexsort((centers[:, 0], centers[:, 1]))
sCenter = centers[sortIndex]

lots = []

#Find parking lots -> Spencer add comments
# Defining the lots and their coordinates
# After sorting, the coordinates of the corners of each lot can be found through an offset.
# sCenter is the 'corners' array
# this code assumes the parking lot will have an even number of parking spots
# i.e the lowest number of lots is two back to back, with a total of 8 coordinates
print("len:",len(sCenter))
for i in range(0, len(sCenter), int(len(sCenter)/2)):
    for j in range(i, i + int(((len(sCenter)/2)-2)/2)):
      lots.append([sCenter[j],sCenter[j+1],sCenter[j+10],sCenter[j+11]])

      c1 = (int(sCenter[j][0]), int(sCenter[j][1]))
      c2 = (int(sCenter[j+1][0]), int(sCenter[j+1][1]))
      c3 = (int(sCenter[j+10][0]), int(sCenter[j+10][1]))
      c4 = (int(sCenter[j+11][0]), int(sCenter[j+11][1]))

      cv2.line(line_image,c1,c2,(255,102,102),2)
      cv2.line(line_image,c1,c3,(255,102,102),2)
      cv2.line(line_image,c3,c4,(255,102,102),2)
      cv2.line(line_image,c2,c4,(255,102,102),2)

finalLots = np.array(lots)

#Parking Space Class -> Paul

#Define Parking Space Class
class ParkingSpace:
    def __init__(self, space, pt1, pt2, pt3, pt4, empty_avg):
        self.space = space
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.pt4 = pt4
        self.empty_avg = empty_avg
    def __str__(self):
        return f'Parking Space: {self.space}, Point 1: {self.pt1}, Point 2: {self.pt2}, Point 3: {self.pt3}, Point 4: {self.pt4}, Empty Average: {self.empty_avg}'

#Create Parking Spaces Objects and Append to a Dictionary
counter = 0
spaces_dict = {}
for parking_spaces in finalLots:
    counter += 1
    pt1 = parking_spaces[0]
    pt2 = parking_spaces[1]
    pt3 = parking_spaces[2]
    pt4 = parking_spaces[3]
    empty_avg = 0
    parking_space = ParkingSpace(counter, pt1, pt2, pt3, pt4, empty_avg)
    spaces_dict[counter] = parking_space

empty_lot = mpimg.imread(path_empty_lot)
empty_lot = empty_lot[:,:,:3]
empty_lot = cv2.cvtColor(empty_lot, cv2.COLOR_BGR2GRAY)

#Find the average intensity of the empty parking spaces
for i in range(1, len(spaces_dict) + 1):
    pt1 = spaces_dict[i].pt1
    pt2 = spaces_dict[i].pt2
    pt3 = spaces_dict[i].pt3
    pt4 = spaces_dict[i].pt4
    
    #Mask image for the parking space
    mask  = np.zeros_like(empty_lot)
    #The corners order actually matters, this orientation is important to make the whole square mask
    space_corners = np.array([[pt4, pt3, pt1, pt2]], dtype=np.int32)

    #Make the mask with the image and the corners
    cv2.fillPoly(mask, space_corners, 255)

    #Process Empty_lot image to prepare for edge detection
    masked_space = cv2.bitwise_and(empty_lot, empty_lot, mask=mask)
    masked_blur = cv2.GaussianBlur(masked_space, (5, 5), 1.1)

    #Detect edges using Canny
    masked_canny = cv2.Canny(masked_blur, 50, 150)

    #Find the average value of the parking spot
    empty_avg = cv2.mean(masked_canny, mask=mask)[0]

    #Add average value to the object
    spaces_dict[i].empty_avg = empty_avg



#Part 2 - Load the image of the parking lot populated with cars to detect vacant spaces
#######################################################################################
#Load parking lot with cars
parking_lot = mpimg.imread(path_cars_lot)
parking_lot = parking_lot.astype(np.uint8)
color_parking_lot = parking_lot
parking_lot = parking_lot[:,:,:3]
parking_lot = cv2.cvtColor(parking_lot, cv2.COLOR_BGR2GRAY)

#Find the average of all the spaces
for i in range(1, len(spaces_dict) + 1):
    pt1 = spaces_dict[i].pt1
    pt2 = spaces_dict[i].pt2
    pt3 = spaces_dict[i].pt3
    pt4 = spaces_dict[i].pt4

    #mask image for the parking space
    mask  = np.zeros_like(parking_lot)
    #the corners order actually matters, this orientation is important to make the whole square mask
    space_corners = np.array([[pt4, pt3, pt1, pt2]], dtype=np.int32)

    #make the mask with the image and the corners
    cv2.fillPoly(mask, space_corners, 255)

    #Apply stuff to prepare for edge detection
    masked_space = cv2.bitwise_and(parking_lot, parking_lot, mask=mask)
    masked_blur = cv2.GaussianBlur(masked_space, (5, 5), 1.1)
    masked_canny = cv2.Canny(masked_blur, 50, 150)

    #find the average of the empty parking space
    new_avg = cv2.mean(masked_canny, mask=mask)[0]

    #if the average of the empty space is less than the average of the occupied space with a 2 int difference (to account for noise variability), then the space is occupied
    if abs((spaces_dict[i].empty_avg - new_avg)) > 2:
        print (f'Parking Space {i} is occupied')

        #Draw a red box around the space
        cv2.line(color_parking_lot,pt4,pt3,(255,0,0),2)
        cv2.line(color_parking_lot,pt3,pt1,(255,0,0),2)
        cv2.line(color_parking_lot,pt1,pt2,(255,0,0),2)
        cv2.line(color_parking_lot,pt2,pt4,(255,0,0),2)
    else:
        #Draw a Green box around the Open Space
        cv2.line(color_parking_lot,pt4,pt3,(0,255,0),2)
        cv2.line(color_parking_lot,pt3,pt1,(0,255,0),2)
        cv2.line(color_parking_lot,pt1,pt2,(0,255,0),2)
        cv2.line(color_parking_lot,pt2,pt4,(0,255,0),2)

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.title('Detected Car(s) in Parking Space')
plt.imshow(color_parking_lot)
plt.savefig('Detected_cars_lot')
plt.show()
        