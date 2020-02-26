import cv2
import numpy as np
import os 
from jaad_data import JAAD

print('hi')
print(os.getcwd())
os.chdir('Data')
jaad_path = os.getcwd()
imdb = JAAD(data_path=jaad_path)
imdb.extract_and_save_images()