# FaceRecogLib
A library for face detection and recognition.
The program uses opencv's haarcascade face filters and hog generation algorythims.
İt is recomended to use this library while creating a database of people to recognise
The project will be further implemented to a real-time face recognition program that uses nvidia's cuda repository.
ust type the names untill it's done, then type "done"
it is recomended to use ubuntu 16.04 with this project. 
The project requires a built in opencv library
My directories are not static therefore any implementation has to be processed via-hard code in any python IDE-environment
The project works succesfully
Questions can be directed to ruzgareserols3(at)gmail.com

IMPORTANT
create three folders in desktop called" ImageData , iSeeYou and FacesOfPeople "
after creating these directories it will be much more easier for the program to find and access to certain files.


How to use
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Run Handler after following the instructions in README(which are really easy to implement). Then, be in front of your camera, write your name when the program asks it(in the prompt). After inputting one's name, the program will automatically capture 400 images in 5-7 seconds. After that the images go under some procedures such as face detection, resizing and hog generation. When you are done taking people into the database, just type "done" to the prompt. A classifier which is a multi-class support vector machine will be generated and trained. The trained version can be found as a function in the testTrainedClassifier class.
  
  



