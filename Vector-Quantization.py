import sys
from PIL import Image as PImage
import numpy as np
from bitarray import bitarray
from matplotlib import pyplot as plt
from numpy import ndarray
from ruamel_yaml import BytesIO
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
    average = np.ceil(average)
    code_book[0] = average - 1
    code_book[1] = average + 1
    start_row = 0
    start_col = -1 * col
    vectors = 2
    n2 = n
    n -= 1
    denominators: ndarray = np.zeros(pow(2, n))
    while n != 0:

        for i in range((arr_row * arr_col) // (row * col)):

            if start_col >= arr_col - col:
                start_col = 0
                start_row += row
            else:
                start_col += col

            temp = arr[start_row: start_row + row, start_col: start_col + col]

            mn_dist = sys.maxsize
            idx = 0
            for j in range(vectors):
                dist = np.abs(temp - code_book[j]).sum()
                if mn_dist > dist:
                    mn_dist = dist
                    idx = j
            denominators[idx] += 1
            ans[idx] += code_book[idx]
        ans = ans // denominators
        code_book_idx = 0

        for k in range(vectors):
            code_book[code_book_idx] = ans[k] + 1
            code_book[code_book_idx + 1] = ans[k] - 1
            code_book_idx += 2
        print(code_book)
        vectors *= 2
        ans = np.zeros((pow(2, n2), row, col))
        n -= 1

    return code_book


def compress(arr, code_book, row, col, n):
    ans = bytearray()
    vectors = pow(2, n)
    arr_row, arr_col = arr.shape

    if arr_row % row != 0:
        arr_row -= arr_row % row
    if arr_col % col != 0:
        arr_col -= arr_col % col
    start_row = 0
    start_col = -1 * col
    for i in range((arr_row * arr_col) // (row * col)):
        code_book_row = -1
        if start_col >= arr_col - col:
            start_col = 0
            start_row += row
        else:
            start_col += col

        temp = arr[start_row: start_row + row, start_col: start_col + col]

        mn_dist = sys.maxsize
        idx = 0
        for j in range(vectors):
            dist = np.abs(temp - code_book[j]).sum()
            if mn_dist > dist:
                mn_dist = dist
                idx = j
        ans.append(idx)

    bits = bytes(ans)
    with open("compressed.txt", "wb") as binary_file:
        binary_file.write(bits)


    return ans


def decompression(arr, row, col, code_book):
    arr_row, arr_col = arr.shape
    if arr_row % row != 0:
        arr_row -= arr_row % row
    if arr_col % col != 0:
        arr_col -= arr_col % col
    start_row = 0
    start_col = -1 * col
    file = open('compressed.txt', 'rb')
    data = file.read()
    file.close()
    data = list(data)
    idx = 0
    decommpressed = []
    for i in range((arr_row * arr_col) // (row * col)):
        decommpressed.append(code_book[int(data[idx])])
        idx += 1
    decommpressed = np.array(decommpressed)
    return decommpressed




imgPath = 'E:\خش برجلك اليمين\Data Comp\Vector-Quantization/photo.png'

img = PImage.open(imgPath).convert("L")

# converts image to numpy array
imgArr = np.asarray(img)
# n = int(input("enter the vector size: "))

# print(type(imgArr), imgArr.shape)
# print(np.min(imgArr),np.max(imgArr)) # 0 to 255 uint8



num = [[1, 2, 7, 9, 4, 11],
       [3, 4, 6, 6, 12, 12],
       [4, 9, 15, 14, 9, 9],
       [10, 10, 20, 18, 8, 8],
       [4, 3, 17, 16, 1, 4],
       [4, 5, 18, 18, 5, 6]]

arr = np.array(num)
# print(sum_submatrices(imgArr, 2, 2))



code_book = get_code_book(imgArr, 2, 2, 2)

ans = compress(imgArr,code_book,2,2,2)
print(imgArr.shape)
imgArr2 = decompression(imgArr,2,2,code_book)
#imgArr = np.squeeze(imgArr, axis=2)
print(imgArr.shape)
savePath = 'decodedImg.png'
print(imgArr2)
#decodedImg = PImage.fromarray(imgArr2)
decodedImg = PImage.fromarray((imgArr2 * 255).astype(np.uint8))
decodedImg.save(savePath) # will save it as gray image
# print(layerR)


