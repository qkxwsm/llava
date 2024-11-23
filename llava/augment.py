import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image

COLOR_MAPPING = {
    'red': (255,0,0),
    'orange': (255,165,0),
    'yellow': (255,255,0),
    'green': (0,255,0),
    'blue': (0,0,255),
    'purple': (128,0,128),
}

def impose_square(img, color):
    color = COLOR_MAPPING[color]

    (m, n, c) = img.shape
    
    max_square_size = 2 * min(m, n) // 4
    min_square_size = min(m, n) // 4
    
    square_size = random.randint(min_square_size, max_square_size)
    
    top_left_x = random.randint(0, m - square_size)
    top_left_y = random.randint(0, n - square_size)
    
    img[top_left_x:top_left_x + square_size, top_left_y:top_left_y + square_size] = (img[top_left_x:top_left_x + square_size, top_left_y:top_left_y + square_size] + color) // 2
    return img

def impose_circle(img, color):
    color = COLOR_MAPPING[color]

    (m, n, c) = img.shape

    max_radius = 2 * min(m, n) // 8  
    min_radius = min(m, n) // 8

    radius = random.randint(min_radius, max_radius)
    # radius = max_radius

    # Randomly choose the center of the circle
    center_x = random.randint(radius, m - radius)
    center_y = random.randint(radius, n - radius)

    for i in range(m):
        # (i - center_x)**2 + (j - center_y)**2 <= radius*radius:
        # (j - center_y)**2 <= radius*radius - (i - center_x)**2
        # j - center_y <= (radius*radius - (i - center_x)**2)**0.5
        # center_y +/- (radius*radius - (i - center_x)**2)**0.5
        width = (radius*radius - (i - center_x)**2)
        if width < 0:
            continue
        width = width**0.5
        low = int(center_y - width)
        high = int(center_y + width)
        img[i, low:high] = (img[i, low:high] + color) // 2

    return img

def impose_triangle(img, color):
    color = COLOR_MAPPING[color]

    (m, n, c) = img.shape

    HEIGHT_TO_BASE = 3**0.5 / 2
    # height = sqrt(3)/2, base = 1
    max_base = int(2 * min(m / HEIGHT_TO_BASE, n)) // 4  
    min_base = int(min(m / HEIGHT_TO_BASE, n)) // 4

    base = random.randint(min_base, max_base)

    top_x = random.randint(0, int(m - base * HEIGHT_TO_BASE))
    top_y = random.randint(0, n - base)
    center_y = top_y + base / 2

    for i in range(top_x, top_x + int(base * HEIGHT_TO_BASE)):   
        dy = int((i-top_x) / (HEIGHT_TO_BASE * 2))
        low = int(center_y-dy)
        high = int(center_y+dy)
        img[i, low:high] = (img[i, low:high] + color) // 2

    return img

def impose_hexagon(img, color):
    color = COLOR_MAPPING[color]

    (m, n, c) = img.shape

    HEIGHT_TO_BASE = 3**0.5 / 2
    max_base = int(2 * min(m / HEIGHT_TO_BASE, n)) // 8
    min_base = int(min(m / HEIGHT_TO_BASE, n)) // 8

    base = random.randint(min_base, max_base) 

    top_x = random.randint(0, int(m - base * HEIGHT_TO_BASE * 2))
    top_y = random.randint(0, int(n - base * 2))

    center_x = int(top_x + base * HEIGHT_TO_BASE)
    center_y = int(top_y + base)

    extended_top_x = center_x - base * HEIGHT_TO_BASE * 2
    for i in range(int(center_x - base * HEIGHT_TO_BASE), center_x):
        dy = (i - extended_top_x) / (HEIGHT_TO_BASE * 2)
        low = int(center_y-dy)
        high = int(center_y+dy)
        img[i, low:high] = (img[i, low:high] + color) // 2
    extended_bottom_x = center_x + base * HEIGHT_TO_BASE * 2

    for i in range(center_x, int(center_x + base * HEIGHT_TO_BASE)):
        dy = (extended_bottom_x - i) / (HEIGHT_TO_BASE * 2)
        low = int(center_y-dy)
        high = int(center_y+dy)
        img[i, low:high] = (img[i, low:high] + color) // 2

    return img  

def impose_shape(img, shape, color):
    if shape == 'triangle':
        img = impose_triangle(img, color)
        
    if shape == 'square':
        img = impose_square(img, color)

    if shape == 'circle':
        img = impose_circle(img, color)

    if shape == 'hexagon':
        img = impose_hexagon(img, color)

    return img

SHAPES = ['triangle', 'square', 'circle', 'hexagon']
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

def augment(path, shape=None, color=None, save_path=None):
    if isinstance(path, str):
        img = Image.open(path)
    else:
        img = path
    img = np.array(img)

    if shape is None:
        shape = random.choice(SHAPES)

    if color is None:
        color = random.choice(COLORS)


    img = impose_shape(img, shape, color)

    if save_path is not None:
        img = Image.fromarray(img)
        img.save(save_path)
    else:
        plt.imshow(img)
        plt.show()

    return img

if __name__ == '__main__':
    PATH = '/Users/davidhu/Downloads/imagenet-mini/train/n02391049/n02391049_10175.JPEG'

    for shape in SHAPES:
        for color in COLORS:
            save_path = f'/Users/davidhu/Downloads/{shape}_{color}.jpeg'
            augment(PATH, shape, color, save_path)
