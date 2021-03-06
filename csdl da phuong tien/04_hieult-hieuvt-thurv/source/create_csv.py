import sys
import os.path

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
# 
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":
    
    BASE_PATH='att_faces'
    SEPARATOR=";"

    label = 0
    with open('at.csv', 'a') as f:
        for dirname, dirnames, filenames in os.walk(BASE_PATH):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    abs_path = "%s/%s" % (subject_path, filename)
                    print ("%s%s%d" % (abs_path, SEPARATOR, label))
                    f.writelines("%s%s%d\n" % (abs_path, SEPARATOR, label))
                label = label + 1