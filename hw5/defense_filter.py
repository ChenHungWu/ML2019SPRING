from PIL import Image
import numpy as np 
import sys

inputDir = sys.argv[1]
if inputDir[-1] != '/' :
    inputDir += '/'
outputDir = sys.argv[2]
if outputDir[-1] != '/' :
    outputDir += '/'

width , height = 224,224


def filter(image , width , height) :
    filterImg = np.zeros(image.shape)
    # RGB
    for c in range(3) :
        for x in range(width) :
            for y in range(height) :
                # 3X3
                neighbors = []
                startI = x - 1
                startJ = y - 1
                for i in range (3) :
                    for j in range(3) :
                        x_n = startI + i 
                        y_n = startJ + j
                        if isValid(x_n,y_n,width,height) :
                            neighbors.append(image[x_n][y_n][c])
                median = getMedian(neighbors)
                filterImg[x][y][c] = median
    filterImg = filterImg.astype(np.uint8)
    return filterImg


    
def isValid(x,y,width,height) :
    Valid = True
    if x < 0 :
        Valid = False
    if x >= width :
        Valid = False

    if y < 0 :
        Valid = False
    if y >= height :
        Valid = False

    return Valid

def getMedian(list_in) :
    list_in.sort()
    if len(list_in) % 2 :
        index = len(list_in) // 2
        return list_in[index]
    else :
        if len(list_in) == 0 :
            print(error)
        else :
            index1 = len(list_in) // 2
            index2 = index1 - 1
            return (list_in[index1]//2 + list_in[index2]//2)




for i in range (200) :
    print('Filter {}-th picture'.format(i+1))
    img = Image.open(inputDir + '{:0>3d}.png'.format(i))
    im = np.array(img)
    imgFilter = filter(im,width,height)
    outputFile = Image.fromarray(imgFilter)
    outputFile.save(outputDir + '{:0>3d}.png'.format(i))

