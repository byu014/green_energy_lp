import itertools
import math
import os
import random
import sys
import numpy as np
import cv2
import uuid
import shutil
import time

def reset_folder(path):
    #  try:
    #      shutil.rmtree(path)
    #  except:
    #       pass
    
    #  time.sleep(1)     
     
     try:
         os.mkdir(path)
     except:
          pass

def warpPerspective(src, M33, sl, gpu):
    if gpu:
        from libs.gpu.GpuWrapper import cudaWarpPerspectiveWrapper
        dst = cudaWarpPerspectiveWrapper(src.astype(np.uint8), M33, (sl, sl), cv2.INTER_CUBIC)
    else:
        dst = cv2.warpPerspective(src, M33, (sl, sl), flags=cv2.INTER_CUBIC)
    return dst

def emboss(img):
    kernel = np.array([
            [-1.5, 0, 0],
            [0, 1.5, 0],
            [0, 0, 0]
        ])
    return cv2.filter2D(img, -1, kernel)

def perspectiveTransform(img):
    # if random.randint(0,5) >= 1:
    #     return img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rotationMatrix = cv2.getRotationMatrix2D((random.randint(0,img.shape[1]), random.randint(0,img.shape[1])), random.randint(-2,2), 1)
    img = cv2.warpAffine(img, rotationMatrix, (img.shape[1], img.shape[0]) ,borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
    warp1 = random.randint(-1,1) * (1 / (1 + math.exp(-random.randint(-20,-8))))
    warp2 = random.randint(-1,1) * (1 / (1 + math.exp(-random.randint(-20,-8))))
    M = np.float32([[1,0,0],
                    [0,1,0],
                    [warp1,warp2,1]])
    img = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])

    return img
    

def invertColor(img,invert=False):
    if invert == False and random.randint(0,20)  >= 5:
        return img
    img = cv2.bitwise_not(img)
    return img

def underline(img):
    if random.randint(0,1) == 0:
        return img
    lineThickness = 2
    img = cv2.line(img, (-10, 80), (img.shape[1]+10, 80), (255,255,255), lineThickness)
    return img

def add_noise(img, strenth = 10):#4
    out = img.astype(np.float64)
    out /= 255.0

    noise_scale = random.randint(0, strenth)/100.0
    out += np.random.normal(scale=noise_scale, size=out.shape)
    out = np.clip(out, 0., 1.)

    out*=255

    out = out.astype(np.uint8)

    return out

def overlay_img(fg, bg, mask, x, y):
    h_fg, w_fg = fg.shape[:2]

    end_x = x + w_fg
    end_y = y + h_fg

    roi = bg[y:end_y, x:end_x]

    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv) ##

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(fg, fg, mask = mask)

    # Put logo in ROI and modify the main image
    try:
       dst = cv2.add(img1_bg, img2_fg)
    except:
       pass

    bg[y:end_y, x:end_x] = dst
    return bg

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[  c, 0.,  s],
                    [ 0., 1., 0.],
                    [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[ 1., 0., 0.],
                    [ 0.,  c, -s],
                    [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[  c, -s, 0.],
                    [  s,  c, 0.],
                    [ 0., 0., 1.]]) * M

    return M

#perspective transfor is better
def make_affine_transform(from_shape, to_shape, 
                        min_scale, max_scale):
    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    M = None
    while True:
        scale = random.uniform(min_scale, max_scale)
                            
        roll =  random.uniform(-0.2, 0.2)  
        pitch = random.uniform(-0.7, 0.7)   
        yaw =   random.uniform(-0.3, 0.3)  

        # Compute a bounding box on the skewed input image (`from_shape`).
        M = euler_to_mat(yaw, pitch, roll)[:2, :2]
        h, w = from_shape[:2]
        corners = np.matrix([[-w, +w, -w, +w],
                                [-h, -h, +h, +h]]) * 0.5
        skewed_size = np.array(np.max(M * corners, axis=1) -
                                np.min(M * corners, axis=1))

        # Set the scale as large as possible such that the skewed and scaled shape
        # is less than or equal to the desired ratio in either dimension.
        scale *= np.min(to_size / skewed_size)

        # Set the translation such that the skewed and scaled image falls within the output shape bounds
        trans = (np.random.random((2,1)) - 0.5)
        trans = ((2.0 * trans) ** 5.0) / 2.0
        if np.any(trans < -0.5) or np.any(trans > 0.5):
            continue
        trans = (to_size - skewed_size * scale) * trans

        center_to = to_size / 2.
        center_from = from_size / 2.

        M = euler_to_mat(yaw, pitch, roll)[:2, :2]
        M *= scale
        M = np.hstack([M, trans + center_to - M * center_from])

        break

    return M

def save_random_img(dir,plate_chars, img):
    #name = dir + plate_chars + "_" + str(uuid.uuid1()) + ".jpg"
    name = os.path.join(dir, str(uuid.uuid1()) + ".jpg")
    cv2.imwrite(name, img)
    return name