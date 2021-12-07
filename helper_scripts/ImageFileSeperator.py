import sys
import os

def seperate(folder, obj_name, illumination_name, training_folder, test_folder):

    if not os.path.exists(training_folder):
        os.mkdir(training_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    for filename in os.listdir(folder):
        print(filename)
        if(filename.find(obj_name) != -1 or filename.find(illumination_name) != -1):
            os.rename(os.path.join(folder, filename), os.path.join(test_folder, filename))
        else:
            os.rename(os.path.join(folder, filename), os.path.join(training_folder, filename))


    

seperate("test","bunny","cambridge", "training", "test")
