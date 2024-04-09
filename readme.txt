Setup and Running the Application
-> The application itself is lotscan.py, it is recommended to use VSCode to view the source code

-> Prior to setup ensure the correct libraries are downloaded on the system
-> Libararies
----> Numpy
----> OpenCV (for Python)
----> matplotlib

-> To run the application there are two options

--> Option A: Using VSCode Run Button
    ---> Press the Run button provided by VSCode, this will trigger the default images to be used, if you want to change the car image you need to specify that in Line 15 by changing the path.
    ---> Options of different images include: images/lot_car.jpg, images/lot_cars.jpg images/lot_cars2.jpg


--> Option B: Using the terminal to run the code and provide arguments indicating the images you want to use/detect cars of
    ---> Note: Ensure you are in the correct directory & the py command will depend on how it is defined in you local environment variables in PATH.
    ---> Command: 'py lotscan.py [Path to empty lot image] [Path to image with cars]'
    ---> Argument 1 [Path to empty lot image]: Refers to the image path where it is the photo of the empty parking lot
    ---> Argument 2 [Path to image with cars]: Refers to the image path where is the photo of the cars to be detected
    ---> Providing no arguments will trigger default images to be used (lot_empty.jpg & lot_cars.jpg)
    ---> Example: 'py lotscan.py images/lot_empty.jpg images/lot_car.jpg'

The program will run, print information about occupied spaces in the terminal and then open an image in matplotlib indicating the occupied spaces. The image of the detected cars will save in the directory as 'Detected_cars_lot.png'