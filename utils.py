from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB, IMREAD_COLOR
from keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np
import random

import glob



def Get_Windows(num,img,img_height,img_width):
    tile_height = int(np.floor(img.shape[0] / num))
    tile_width = int(np.floor(img.shape[1] / num))

    windows = []
    for r in range(0, img.shape[0] - tile_height+1, tile_height):
        for c in range(0, img.shape[1] - tile_width+1, tile_width):
            img_tile = img[r:r + tile_height, c:c + tile_width]
            img_tile = cvtColor(resize(img_tile, (img_height, img_width)), COLOR_BGR2RGB).astype('float32')
            img_tile = np.expand_dims(img_tile, axis=0)
            img_tile = preprocess_input(img_tile)
            windows.append(img_tile)
    return windows


def Get_Tiles(img,img_height,img_width,kernel_size,stride,mpratio):

    tile_height=int(np.floor(kernel_size/mpratio))
    tile_width=tile_height
    stride_length=int(np.floor(stride/mpratio))

    tiles=[]
    for r in range(0,img.shape[0]-stride_length,stride_length):
        for c in range(0, img.shape[1] - stride_length, stride_length):
            if r+tile_height<=img.shape[0] and c+tile_width<=img.shape[1]:
                img_tile=img[r:r+tile_height,c:c+tile_width]
            elif r + tile_height <= img.shape[0] and not c + tile_width <= img.shape[1]:
                img_tile = img[r:r + tile_height, img.shape[1] - tile_width:img.shape[1]]
            elif not r + tile_height <= img.shape[0] and c + tile_width <= img.shape[1]:
                img_tile = img[img.shape[0] - tile_height:img.shape[0], c:c + tile_width]
            elif not r + tile_height <= img.shape[0] and not c + tile_width <= img.shape[1]:
                img_tile = img[img.shape[0] - tile_height:img.shape[0], img.shape[1] - tile_width:img.shape[1]]

            img_tile = cvtColor(resize(img_tile, (img_height, img_width)), COLOR_BGR2RGB).astype('float32')
            img_tile = np.expand_dims(img_tile, axis=0)
            img_tile = preprocess_input(img_tile)
            tiles.append(img_tile)

    tiles=np.vstack(tiles)
    return tiles


def Get_Bottleneck_Tiles_Unlabeled(folder,mpratio):

    """Collect a batch of images from data set and pass through ResNet
    to get bottleneck features. Return codes being bottleneck features and labels being label. This
    version does tiling of each tile to get better feature detection at various magnifications"""

    # Load ResNet50
    model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

    img_height, img_width = 224, 224

    fraction=1
    codes=None
    file_id=[]

    print("Starting {} images".format(type))
    files = glob.glob(folder + '*.tif')
    files = random.sample(files, int(fraction * len(files)))
    for ii,file in enumerate(files,1):
        img=imread(file)

        kernel_size, stride = 100, 50  # height and stride of kernel in terms of microns
        tiles = Get_Tiles(img, img_height, img_width, kernel_size, stride, mpratio)
        codes_tiles = model.predict_on_batch(tiles)
        code_1 = np.amax(codes_tiles, axis=0).reshape(1, -1)

        kernel_size, stride = 400, 200
        tiles = Get_Tiles(img, img_height, img_width, kernel_size, stride, mpratio)
        codes_tiles = model.predict_on_batch(tiles)
        code_2 = np.amax(codes_tiles, axis=0).reshape(1, -1)

        code = np.concatenate((code_1, code_2), axis=0)
        code = np.expand_dims(code, axis=0)
        file_id.append(file)

        if codes is None:
            codes=code
        else:
            codes=np.concatenate((codes,code),axis=0)

        print('{} images processed'.format(ii))

    return codes,file_id






























