import os
import cv2
from ultralytics import YOLO
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model
import tensorflow as tf
from utils import *
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import shutil




def read_pictures_from_folder(folder_path):
    picture_list = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError("The specified folder does not exist.")

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        # Check if the file is an image
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the image using OpenCV
                image = cv2.imread(file_path)
                picture_list.append(image)
            except Exception as e:
                print(f"Error while processing {file_name}: {e}")

    return picture_list, file_list


def locate_sayac(model, image_path, output_folder_path, image_name):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if os.path.exists(os.path.join(output_folder_path,image_name)):
        shutil.rmtree(os.path.join(output_folder_path,image_name))


    model.predict(image_path,save_txt = True, project=output_folder_path, name=image_name)
    
    
def parseTxt(label_path, target):  #target:1 for sayac, 0 for endeks
    #print("label path is: " + label_path)
    with open(label_path, "r") as file:
        lines = file.readlines()
        print(lines)
    for line in lines:
        parts = line.split(" ")
        #print(parts)
     
        if(int(parts[0])==target):
            
                return float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])

           
                
    for line in lines:
        parts = line.split(" ")
        #print(parts)
     
        if(int(parts[0])!=target):
            
                return float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
        



def crop_sayac(image_path, label_path, target):
    image = Image.open(image_path)
    image_width = image.width
    image_height = image.height

    center_x,center_y,width,height = parseTxt(label_path, target)

    center_x = center_x * image_width
    center_y = center_y * image_height
    width = width * image_width
    height = height * image_height

    # Calculate the coordinates for cropping
    left = center_x - width // 2
    upper = center_y - height // 2
    right = center_x + width // 2
    lower = center_y + height // 2

    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))
    new_dir_name = ( "C:\\Users\\anile\\Desktop\\pipeline_sayac\\cropped\\"+str( os.path.basename(image_path) ) ).replace(".jpeg","")
    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)
    cropped_image.save(new_dir_name+"\\"+str( os.path.basename(image_path) ))
    # Save or display the cropped image
    cropped_image.show()  # Display the cropped image

    test_filenames = [
        new_dir_name+"\\"+str( os.path.basename(image_path) )
        ]
    num_images = 1
        
    return test_filenames

    

    print("save path is: ", save_path)
    locate_sayac(model_yolo,save_path,"C:\\Users\\anile\\Desktop\\pipeline_sayac\\res2",os.path.basename(save_path))
    #exit(0)
    # Save the prediction results
    


#results = model_yolo.predict(source=pictures, save=True, save_txt=True, line_width=1, show_conf=False)  # save predictions as labels

def close_window():
    #root.destroy()  # Terminate the Tk instance and close the window
    exit(0)

root = tk.Tk()
root.title("Image Selector")
root.protocol("WM_DELETE_WINDOW", close_window)

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if file_path:
        # Display the selected image in the window
        img = Image.open(file_path)
        img.thumbnail((400, 400))  # Resize image to fit in the window
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk

        img_name = os.path.basename(file_path)

        
        locate_sayac(model_yolo,file_path,"C:\\Users\\anile\\Desktop\\pipeline_sayac\\predictions", img_name)
        img_name_without_type = img_name.replace(".jpeg", "")

        
        label_path = "C:\\Users\\anile\\Desktop\\pipeline_sayac\\predictions\\" +img_name + "\\labels\\" +img_name_without_type+ ".txt"
        test_filenames = crop_sayac(file_path, label_path,1)
        
        save_path=display_examples(
        model,
        test_filenames,
        num_images=1,
        size=(224, 224),
        crop_center=True,
        crop_largest_rect=True,
        preprocess_func=preprocess_input,
        
        )

        save_path2=display_examples(
        model,
        [save_path],
        num_images=1,
        size=(224, 224),
        crop_center=True,
        crop_largest_rect=True,
        preprocess_func=preprocess_input,
        
        )
        
        save_path3=display_examples(
        model,
        [save_path2],
        num_images=1,
        size=(224, 224),
        crop_center=True,
        crop_largest_rect=True,
        preprocess_func=preprocess_input,
        
        )
        
        print("img at: ", save_path3)

        locate_sayac(model_yolo,save_path3,"C:\\Users\\anile\\Desktop\\pipeline_sayac\\res2", img_name)
        img_name_without_type = img_name.replace(".jpeg", "")
        label_path = "C:\\Users\\anile\\Desktop\\pipeline_sayac\\res2\\" +img_name + "\\labels\\" +img_name_without_type+ ".txt"
        test_filenames = crop_sayac(save_path, label_path,0)


        # Perform your operation on the selected image here
        # For example: prediction = model_yolo.predict(file_path)

        


model_yolo = YOLO("C:\\Users\\anile\\Desktop\\pipeline_sayac\\models\\best.pt")

tf.compat.v1.enable_eager_execution()
model_location = "C:\\Users\\anile\\Desktop\\pipeline_sayac\\models\\rotnet_street_view_resnet50.hdf5"
model = load_model(model_location, custom_objects={'angle_error': angle_error})
tf.config.run_functions_eagerly(True)
model.run_eagerly = True

# Button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

# Label to display the selected image
label = tk.Label(root)
label.pack()

# Start the main GUI event loop
root.mainloop()
