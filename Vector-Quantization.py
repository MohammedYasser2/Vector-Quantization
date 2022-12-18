from PIL import Image as PImage
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def average_matrix(arr, row, col):

    ans = [[0] * row for i in range(col)]
    ans = np.array(ans, dtype=object)

    arr_row, arr_col = arr.shape
    if col is None:
        col = row

    if arr_row % row != 0:
        arr_row -= arr_row % row
    if arr_col % col != 0:
        arr_col -= arr_col % col

    for i in range(arr_row):
        for j in range(arr_col):
            ans[i % row][j % col] += arr[i][j]

    ans = ans / ((arr_row * arr_col) / (row * col))
    return ans

def get_code_book(arr, row, col, n):

    average = average_matrix(arr, row, col)
    arr_row, arr_col = arr.shape
    if arr_row % row != 0:
        arr_row -= arr_row % row
    if arr_col % col != 0:
        arr_col -= arr_col % col

    code_book = np.zeros((pow(2, n), row, col))
    ans = np.zeros((pow(2, n), row, col))
    code_book[0] = np.floor(average - 1)
    code_book[1] = np.floor(average + 1)
    start = 1
    end = 1
    vectors = 2
    n2 = n
    n -= 1
    while n != 0:

       for i in range((arr_row * arr_col) // (row * col)):

           temp = arr[start * row][end * col]
           mn_dist = 255
           idx = 0
           for j in range(vectors):
              dist = np.linalg.norm(code_book[j] - temp)
              if mn_dist > dist:
                  mn_dist = dist
                  idx = j
           ans[idx] += code_book[idx]
       ans -= 1
       vectors += 2
       ans = ans / 2
       code_book = ans
       ans = np.zeros((pow(2, n2), row, col))
       n -= 1


    return code_book









imgPath = 'E:\خش برجلك اليمين\Data Comp\Vector-Quantization/photo.png'


img = PImage.open(imgPath).convert("L")




# converts image to numpy array
imgArr = np.asarray(img)
#n = int(input("enter the vector size: "))

#print(type(imgArr), imgArr.shape)
#print(np.min(imgArr),np.max(imgArr)) # 0 to 255 uint8
#savePath= 'decodedImg.png'
#decodedImg = PImage.fromarray(imgArr)
#decodedImg.save(savePath) # will save it as gray image


num = [[1,2,7,9,4,11],
       [3,4,6,6,12,12],
       [4,9,15,14,9,9],
       [10,10,20,18,8,8],
       [4,3,17,16,1,4],
       [4,5,18,18,5,6]]

arr = np.array(num)

#print(sum_submatrices(imgArr, 2, 2))


# calculating Euclidean distance
# using linalg.norm()
print(get_code_book(arr,2,2,2))

#print(layerR)