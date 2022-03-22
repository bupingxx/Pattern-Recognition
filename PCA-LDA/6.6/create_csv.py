import sys
import os.path
import cv2 as cv

if __name__ == "__main__":

    BASE_PATH="ORL"
    SEPARATOR=";"
    
    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print (abs_path, SEPARATOR, label)
            label = label + 1