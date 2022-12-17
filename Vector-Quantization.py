# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:43:11 2022

@author: Salah
"""


from PIL import Image
import numpy as np

imgPath = 'E:\خش برجلك اليمين\Data Comp\Vector-Quantization/photo.png'
img = Image.open(imgPath).convert("L")
print(type(img))
# converts image to numpy array
imgArr = np.asarray(img)
print(imgArr)
print(type(imgArr), imgArr.shape)
print(np.min(imgArr),np.max(imgArr)) # 0 to 255 uint8
savePath = 'something.png'
decodedImg = Image.fromarray(imgArr)

decodedImg.save(savePath) # will save it as gray image

