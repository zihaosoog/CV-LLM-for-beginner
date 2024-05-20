import string, os
#import os
import pdb
from PIL import Image


def find_x(bbox_left,image_width,box_xmax):
	image_width = 1.0 * int(image_width)
	absolute_x = int(bbox_left) + 0.5 * (int(box_xmax) - int(bbox_left))
	x = absolute_x / image_width
	return str(x)

	

def find_y(image_height,box_ymax,bbox_top):
	image_height = 1.0 * int(image_height)
	absolute_y = int(bbox_top) + 0.5 * (int(box_ymax) - int(bbox_top))
	y = absolute_y / image_height
	return str(y)

def find_width(box_xmax,bbox_left,image_width): 
	absolute_width = int(box_xmax) - int(bbox_left)
	image_width = 1.0 * int(image_width)
	width = absolute_width / image_width 
	return str(width)


def find_height(bbox_top,box_ymax,image_height):
	absolute_height = int(box_ymax) - int(bbox_top)
	image_height = 1.0 * int(image_height)
	height = absolute_height / image_height
	return str(height)

label_path= "./VisDrone2019-DET-train/labels/"
file_default = "./VisDrone2019-DET-train/annotations/"
image_default= "./VisDrone2019-DET-train/images/"


# label_path= "./VisDrone2019-DET-val/labels/"
# file_default = "./VisDrone2019-DET-val/annotations/"
# image_default= "./VisDrone2019-DET-val/images/"

if not os.path.exists(label_path):
    print(os.path.exists(label_path))
    os.makedirs(label_path)

#f = "./"
#f = "./"
for f in os.listdir(file_default):

    if (f != "Visdrone2YOLO.py"):
        print(f)
        fname = file_default + f
        fname_out = label_path + f
        filename_without_ext =  str(f.split(".")[0])
        
        imagepath = image_default + filename_without_ext + ".jpg"
                
        #fname = filename
        #fname_out = filename_without_ext + "_out.txt"

        im = Image.open(imagepath)
        image_width, image_height = im.size
        print(image_width, image_height)
#        image_width = 1920
#        image_height = 1080

        content = []

        with open(fname) as f1:
            content = f1.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content] 

        new_content = []

        for x in content:
            y = x.split(",")
            #del y[4]
                #del y[6]
            #del y[7]


            bbox_left = y[0]
            bbox_top = y[1]
            bbox_width = y[2]
            bbox_height = y[3]
            box_xmax = str(int(y[0]) + int(y[2]))
            box_ymax = str(int(y[1]) + int(y[3]))



            place_0_value = y[5]
            place_1_value = find_x(bbox_left,image_width,box_xmax)
            place_2_value = find_y(image_height,box_ymax,bbox_top)
            place_3_value = find_width(box_xmax,bbox_left,image_width)
            place_4_value = find_height(bbox_top,box_ymax,image_height)

            output = str(place_0_value) + " " + str(place_1_value) + " " + str(place_2_value) + " " + str(place_3_value) + " " + str(place_4_value)
            new_content.append(output)

        

        with open(fname_out, 'w') as f2:
            f2.truncate(0)
            for item in new_content:
                f2.write("%s\n" % item)