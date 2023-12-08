#!/usr/bin/env python
#
# picmerj.py 
#
# combine 2 input images into a 3rd --> the output image
#
# TO DO:
# 1. resize image to larger of pair
# 2. green screen
# 3. provide offset for smaller image within larger
# 4. masking
#
# 0. go see Joanna & band https://www.youtube.com/watch?v=iSwoM7WIy5M
#
import os

from PIL import Image
from PIL import ImageFilter
import sys
import random
import cv2

white = (255,255,255)
black = (0,0,0)
max_red = (255,0,0)
max_green = (0,255,0)
max_blue = (0,0,255)


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    # 2DO: preserve aspect ratio
    cv2.resizeWindow(name_of_window, 403*3, 302*3)
    cv2.moveWindow(name_of_window, 100, 50)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color(x,y):
    red = (x*x+y*y)%255
    green = (y*y*2+2*x*x)%255
    blue = (y*y*3+3*x*x)%255
    return (red, green, blue)

    
def noise_or_color(col, noise):
    """higher value for noise (a float in [0..1]) means more noise"""
    if random.random() < noise:
        red = (col[0] + random.randrange(-50,50)) % 256
        green = (col[1] + random.randrange(-50,50)) % 256
        blue = (col[2] + random.randrange(-50,50)) % 256
        return (red,green,blue)
    else:
        return col

        
def clamp(val,min,max):
    if (val < min): return min
    if (val > max): return max
    return val

        
def noisy_color(col, noise, amount):
    """higher value for noise (a float in [0..1]) means more noise"""
    if random.random() < noise:
        red = (col[0] + random.randrange(-amount,amount))
        green = (col[1] + random.randrange(-amount,amount))
        blue = (col[2] + random.randrange(-amount,amount))
        red = clamp(red,0,255)
        green = clamp(green,0,255)
        blue = clamp(blue,0,255)
        return (red,green,blue)
    else:
        return col

               
def sum_channels(pixel):
    return pixel[0] + pixel[1] + pixel[2]

    
def red_channel(pixel):
    return pixel[0]

    
def green_channel(pixel):
    return pixel[1]

    
def blue_channel(pixel):
    return pixel[2]

    
def euclidean_dist(px,px2):
    return (px[0] - px2[0])**2 + (px[1] - px2[1])**2 + (px[2] - px2[2])**2
    

def dist_to_target(px):
    return euclidean_dist(px, (target_red, target_green, target_blue))
    

def average(px1, px2):
    return (int((px1[0] + px2[0])/2), int((px1[1] + px2[1])/2), int((px1[2] + px2[2])/2))


def random_source(px1, px2):
    if (random.random() < 0.5):
        return px1
    return px2
    

def brighter(px1, px2):
    if (sum_channels(px1) > sum_channels(px2)):
        return px1
    return px2
    

def darker(px1, px2):
    if (sum_channels(px1) < sum_channels(px2)):
        return px1
    return px2
    

def redder(px1, px2):
    if (red_channel(px1) > red_channel(px2)):
        return px1
    return px2
    

def greener(px1, px2):
    if (green_channel(px1) > green_channel(px2)):
        return px1
    return px2
    

def bluer(px1, px2):
    if (blue_channel(px1) > blue_channel(px2)):
        return px1
    return px2
    

def less_red(px1, px2):
    if (red_channel(px1) < red_channel(px2)):
        return px1
    return px2    
    

def less_green(px1, px2):
    if (green_channel(px1) < green_channel(px2)):
        return px1
    return px2    
    

def less_blue(px1, px2):
    if (blue_channel(px1) < blue_channel(px2)):
        return px1
    return px2    
    

def redder2(px1, px2):
    if (red_channel(px1) - (blue_channel(px1) + green_channel(px1)) > red_channel(px2) - (blue_channel(px2) + green_channel(px2))):
        return px1
    return px2
    

def greener2(px1, px2):
    if (green_channel(px1) - (red_channel(px1) + blue_channel(px1)) > green_channel(px2) - (red_channel(px2) + blue_channel(px2))):
        return px1
    return px2


def bluer2(px1, px2):
    if (blue_channel(px1) - (red_channel(px1) + green_channel(px1)) > blue_channel(px2) - (red_channel(px2) + green_channel(px2))):
        return px1
    return px2
    

def closer(px1, px2):
    if (dist_to_target(px1) < dist_to_target(px2)):
        return px1
    return px2
    

def further(px1, px2):
    if (dist_to_target(px1) > dist_to_target(px2)):
        return px1
    return px2
    

def below(px1, px2):
    if (red_channel(px1) < target_red and green_channel(px1) < target_green and blue_channel(px1) < target_blue):
        return px2
    return px1
       

def above(px1, px2):
    if (red_channel(px1) > target_red and green_channel(px1) > target_green and blue_channel(px1) > target_blue):
        return px2
    return px1
       

def bluer_than_threshold(px1, px2):
    if (bluer(px2, (target_red, target_green, target_blue)) == px2):
        return px2
    return px1
    

def p1_if_p2_dark(px1, px2):
    if (sum_channels(px2) < darkness_threshold):
        return px1
    return px2
    

def combine(im1, im2, width, height, combination, out_img):
    for i in range(width):
        for j in range(height):
            out_img[i,j] = combination(im1[i,j], im2[i,j])
            

def checkerboard(im1, im2, width, height, out_img):
    for i in range(width):
        if (i % 2) == 0:
            for j in range(height):
                if (j % 2) == 0:
                    out_img[i,j] = im1[i,j]
                else:
                    out_img[i,j] = im2[i,j]
        else:
            for j in range(height):
                if (j % 2) == 1:
                    out_img[i,j] = im1[i,j]
                else:
                    out_img[i,j] = im2[i,j]
        

def chessboard(im1, im2, width, height, s1, s2, out_img):
    for i in range(width):
        if (i % s1) < s2:
            for j in range(height):
                if (j % s1) < s2:
                    out_img[i,j] = im1[i,j]
                else:
                    out_img[i,j] = im2[i,j]
        else:
            for j in range(height):
                if (j % s1) >= s2:
                    out_img[i,j] = im1[i,j]
                else:
                    out_img[i,j] = im2[i,j]
        
            
if __name__ == '__main__':

    try:
        print( 'trying to open ' + sys.argv[1] )
        img = Image.open(sys.argv[1]).convert('RGB')
        print( 'trying to open ' + sys.argv[2] )
        img2 = Image.open(sys.argv[2]).convert('RGB')
    except:
        print( 'failed to open input file!' )
        img_size = (1000, 1000)
        img = Image.new('RGB', img_size, white)
        try:
            img2 = Image.open('images/seven11.png').convert('RGB')
        except:
            exit()

    # 2DO: preserve aspect ratio in case of resize
    (x_pixels, y_pixels) = img.size
    (width, height) = img2.size
    # try to prevent out-of-range errors 
    # by using the smaller dimensions
    if (img.size[0] < img2.size[0]): 
        width = x_pixels
        height = y_pixels
        print('shrinking img2')
        img3 = img2.resize( (width, height) )
        img3.save('test.jpg')
        img2 = img3
    else:
        print('shrinking img')
        img3 = img.resize((width, height))
        img3.save('test.jpg')
        img = img3

    img_pix = img.load()
    img2_pix = img2.load()
	
    new_image = Image.new(img.mode, (width, height))
    print( 'output image will be', width, 'by', height )
    print( 'image1 is', img.size[0], 'by', img.size[1] )
    print( 'image2 is', img2.size[0], 'by', img2.size[1] )
    
    target_red = darkness_threshold = 11
    target_green = 111
    target_blue = 255
    try:
        target_red = darkness_threshold = int(sys.argv[4])
        target_green = int(sys.argv[5])
        target_blue = int(sys.argv[6])
    except:
        pass
        
    print( target_red, target_green, target_blue )

    try:
        combiner_code = sys.argv[3]
    except:
        combiner_code = 'l'

    combiners = {}
    combiners['a'] = average
    combiners['b'] = bluer
    combiners['b2'] = bluer2
    combiners['bt'] = bluer_than_threshold
    combiners['c'] = closer
    combiners['cb'] = checkerboard
    combiners['d'] = darker
    combiners['d2'] = p1_if_p2_dark # else p2, presumably
    combiners['f'] = further
    combiners['g'] = greener
    combiners['g2'] = greener2
    combiners['l'] = brighter
    combiners['lb'] = less_blue
    combiners['lg'] = less_green
    combiners['lr'] = less_red
    combiners['r'] = redder
    combiners['r2'] = redder2
    combiners['ra'] = above
    combiners['rb'] = below
    combiners['rs'] = random_source
    combiners['xb'] = chessboard

    if combiner_code in combiners:
        combiner = combiners[ combiner_code ]
    else:
        print( 'combiner_code "', combiner_code, '" not recognized, defaulting to "l" (lighter/brighter)' )
        combiner = brighter  # always [default to] the bright side of life [whistling]
        
    print( 'combiner_code is [', combiner_code, ']')
        
    if combiner_code == 'cb':
        combiner( img_pix, img2_pix, width, height, new_image.load() )
    elif combiner_code == 'xb':
        # 2DO: better names than target_xxxx
        combiner( img_pix, img2_pix, width, height, target_red, target_green, new_image.load() )
    else:
        print('e.g.:', combiner(img_pix[0,0], img2_pix[0,0]), '<==combiner(', img_pix[0,0], ',', img2_pix[0,0], ')')
        combine( img_pix, img2_pix, width, height, combiner, new_image.load() )
    
    newfile = ''
    try:
        left_part = sys.argv[1].split('/')[-1]
        left_part = left_part[:-4]
        if left_part[1] == ':': left_part = left_part[2:]
        right_part = sys.argv[2].split('/')[-1]
        right_part = right_part[:-4]
        if right_part[1] == ':': right_part = right_part[2:]
        newfile = 'newimages/' + left_part + '-' + right_part + '-' + combiner_code + '.jpg'
        print( 'writing to ' + newfile )
        new_image.save(newfile)
    except:
        print( 'writing to default.jpg' )
        new_image.save('default.jpg')
    
    if newfile != '' and 'ix' not in os.name:
        image = cv2.imread(newfile)
        viewImage(image, newfile)
