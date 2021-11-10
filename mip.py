#! /usr/bin/python3

# mip.py
#
# MacGinitie's Image Processor

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import os
import random
import sys
import cv2

white = 255, 255, 255
black = 0, 0, 0


def view_image(name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    # 2DO: preserve aspect ratio
    cv2.resizeWindow(name_of_window, 403 * 3, 302 * 3)
    cv2.moveWindow(name_of_window, 100, 50)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_rgb_colors(filename):
    clist = []
    c = open(filename).readlines()
    for l in c:
        (r, g, b) = l.strip().split(' ')
        clist.append((r, g, b))
    return clist


def color(x, y):
    red = (x * x + y * y) % 255
    green = (y * y * 2 + 2 * x * x) % 255
    blue = (y * y * 3 + 3 * x * x) % 255
    return red, green, blue


def noise_or_color(col, noise):
    """higher value for noise (a float in [0..1]) means more noise"""
    if random.random() < noise:
        red = (col[0] + random.randrange(-50, 50)) % 256
        green = (col[1] + random.randrange(-50, 50)) % 256
        blue = (col[2] + random.randrange(-50, 50)) % 256
        return red, green, blue
    else:
        return col


def clamp(val, mini, maxi):
    if val < mini:
        return mini
    if val > maxi:
        return maxi
    return val


def noisy_color(col, noise, amount):
    """higher value for noise (a float in [0..1]) means more noise"""
    if random.random() < noise:
        red = (col[0] + random.randrange(-amount, amount))
        green = (col[1] + random.randrange(-amount, amount))
        blue = (col[2] + random.randrange(-amount, amount))
        red = clamp(red, 0, 255)
        green = clamp(green, 0, 255)
        blue = clamp(blue, 0, 255)
        return red, green, blue
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


def euclidean_dist(px, px2):
    return (px[0] - px2[0]) ** 2 + (px[1] - px2[1]) ** 2 + (px[2] - px2[2]) ** 2


def greyscale_contrast(px1, px2):
    return abs(sum_channels(px1) - sum_channels(px2))


def is_high_greyscale_contrast(px1, px2, paramlist):
    thresh = paramlist[0] if len(paramlist) > 0 else 10
    return greyscale_contrast(px1, px2) > thresh


def is_high_contrast(px1, px2, paramlist):
    thresh = paramlist[0] if len(paramlist) > 0 else 10
    return euclidean_dist(px1, px2) > (thresh * thresh)


def is_edge(pixel, neighbors, diffcalc):
    for nbrpxl in neighbors:
        if diffcalc(nbrpxl, pixel):
            return True
    return False


def is_black(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0


def is_white(pixel):
    return pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255


def is_above_darkness_threshold(pixel, darkness_threshold):
    return sum_channels(pixel) > darkness_threshold


def is_lone_black(pixel, neighbors):
    if is_black(pixel):
        for nbrpxl in neighbors:
            if is_black(nbrpxl):
                return False
        return True
    return False


def dither(pixel, paramlist=[]):
    # smaller values of lightness should yield lighter images
    lightness = paramlist[0] if len(paramlist) > 0 else 255
    if sum_channels(pixel) < (3 * random.randrange(0, lightness)):
        return black
    return white


def diag_stripe(x, y, modulus):
    if (x - y // 2) % modulus == 0:
        return black
    return white


def pseudo_crosshatch(pixel, x, y, paramlist=[]):
    threshold1 = paramlist[0] if len(paramlist) > 0 else 100
    threshold2 = paramlist[1] if len(paramlist) > 1 else 100 + threshold1
    threshold3 = paramlist[2] if len(paramlist) > 2 else 100 + threshold2
    sumchan = sum_channels(pixel)
    if sumchan < threshold1:
        return black
    if sumchan < threshold2:
        return diag_stripe(x, y, 5)
    if sumchan < threshold2:
        return diag_stripe(x, y, 9)
    return white


def auto_crosshatch(pixel, x, y, paramlist=[]):
    sumchan = sum_channels(pixel)
    if sumchan < 50:
        return black
    modulus = 2
    thresholds = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    for thr in thresholds:
        if sumchan < thr:
            return diag_stripe(x, y, modulus)
        modulus += 2
    return white


def map_value_to_level(value):
    if value < 26:
        return 0
    elif value < 52:
        return 1
    elif value < 79:
        return 2
    elif value < 110:
        return 3
    elif value < 135:
        return 4
    elif value < 160:
        return 5
    elif value < 185:
        return 6
    elif value < 210:
        return 7
    else:
        return 8


def load_hatches(num_shades):
    hlist = []
    for i in range(num_shades):
        # fname = "images/hatch" + str(i) + ".png"
        fname = "images/hatch" + str(i) + ".bmp"
        imgo = Image.open(fname)
        hlist.append(imgo.load())
    return hlist


def nbr_is_black(pixel, neighbors, thresh, dot_size):
    greyval = sum_channels(pixel)
    if greyval < thresh:
        return True
    if dot_size == 2:
        percent = 0.8
    elif dot_size == 3:
        percent = 0.6
    elif dot_size == 4:
        percent = 0.4
    else:
        percent = 0.99
    count = 0
    for nbrpxl in neighbors:
        if sum_channels(nbrpxl) < thresh:
            count += 1
    if count > 0:
        if random.random() > percent:
            return True
    return False


# originally imported from sketcher.py, but adapted for use here in mip.py    
def maybe_black_dot(pixel, neighborhood, paramlist=[]):
    greyval = sum_channels(pixel)
    thresh = 200 if len(paramlist) < 1 else paramlist[0]
    dot_size = 2 if len(paramlist) < 2 else paramlist[1]
    if nbr_is_black(pixel, neighborhood, thresh, dot_size):
        return black
    else:
        return white


def remove_lone_black_pixels(pixel, neighborhood, param):
    if is_lone_black(pixel, neighborhood):
        return white
    return pixel


def scramble(pixel, neighborhood, param):
    index = random.randrange(0, len(neighborhood))
    return neighborhood[index]


def contrast_map(pixel, neighborhood, param):
    divisor = param[0] if len(param) > 0 else 6100
    contrast = 0
    for px in neighborhood:
        contrast += euclidean_dist(pixel, px)
    if divisor != 0:
        contrast /= divisor
    contrast = 255 - int(contrast)
    return contrast, contrast, contrast


def edge_convolve(pixel, neighborhood, param):
    dc = lambda px1, px2: is_high_greyscale_contrast(px1, px2, param)
    if is_edge(pixel, neighborhood, dc):
        return 0, 0, 0
    return 255, 255, 255


def edge_convolve2(pixel, neighborhood, param):
    dc = lambda px1, px2: is_high_contrast(px1, px2, param)
    if is_edge(pixel, neighborhood, dc):
        return 0, 0, 0
    return 255, 255, 255


def edge_convolve3(pixel, neighborhood, param):
    dc = lambda px1, px2: is_high_greyscale_contrast(px1, px2, param)
    if is_edge(pixel, neighborhood, dc):
        return 0, 0, 0
    return pixel


def edge_convolve4(pixel, neighborhood, param):
    dc = lambda px1, px2: is_high_contrast(px1, px2, param)
    if is_edge(pixel, neighborhood, dc):
        return 0, 0, 0
    return pixel


def edge_convolve5(pixel, neighborhood, param):
    dc = lambda px1, px2: is_high_contrast(px1, px2, param)
    if is_edge(pixel, neighborhood, dc):
        return max(pixel[0] - 11, 0), max(pixel[1] - 11, 0), max(pixel[2] - 11, 0)
    return min(pixel[0] + 11, 255), min(pixel[1] + 11, 255), min(pixel[2] + 11, 255)


def random_color():
    return random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)


def random_dark_color():
    return random.randrange(0, 125), random.randrange(0, 125), random.randrange(0, 125)


def random_light_color():
    return random.randrange(100, 256), random.randrange(100, 256), random.randrange(100, 256)


def random_color_edge_convolve(pixel, neighborhood, param):
    dc = lambda px1, px2: is_high_contrast(px1, px2, param)
    if is_edge(pixel, neighborhood, dc):
        return random_color()
    return 0, 0, 0


def darken(pixel, x, y, paramlist=[]):
    darkness_threshold = paramlist[0] if len(paramlist) > 0 else 20
    return max(0, pixel[0] - darkness_threshold), max(0, pixel[1] - darkness_threshold), max(0, pixel[
        2] - darkness_threshold)


# this is similar to whiten()    
def lighten(pixel, x, y, paramlist=[]):
    pct_increase = paramlist[0] if len(paramlist) > 0 else 10
    (r, g, b) = pixel
    r = min(int(r * (100 + pct_increase) / 100.), 255)
    g = min(int(g * (100 + pct_increase) / 100.), 255)
    b = min(int(b * (100 + pct_increase) / 100.), 255)
    return r, g, b


def negate(pixel, x, y, paramlist):
    return 255 - pixel[0], 255 - pixel[1], 255 - pixel[2]


def blacken_nonwhite(pixel, x, y, paramlist):
    if is_white(pixel):
        return white
    return black


def blacken_below_threshold(pixel, x, y, paramlist=[]):
    thresh = 700 if len(paramlist) == 0 else paramlist[0]
    if sum_channels(pixel) > thresh:
        return pixel
    return black


def black_or_white(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if is_above_darkness_threshold(pixel, thresh):
        return white
    return black


def blue_tint(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if is_above_darkness_threshold(pixel, thresh):
        return pixel[0] // 2, pixel[1] // 2, 255
    return pixel


def light_blue_tint(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if is_above_darkness_threshold(pixel, thresh):
        return int(pixel[0] * 0.8), int(pixel[1] * 0.8), 255
    return pixel


def enrich(pixel, x, y, paramlist):
    biggest = max(pixel[0], pixel[1], pixel[2])
    if biggest == pixel[0]:
        return 255, pixel[1], pixel[2]
    elif biggest == pixel[1]:
        return pixel[0], 255, pixel[2]
    return pixel[0], pixel[1], 255


def enrich2(pixel, x, y, paramlist=[]):
    darkness_threshold = paramlist[0] if len(paramlist) > 0 else 10
    biggest = max(pixel[0], pixel[1], pixel[2])
    if biggest == pixel[0]:
        return min(255, pixel[0] + darkness_threshold), pixel[1], pixel[2]
    elif biggest == pixel[1]:
        return pixel[0], min(255, pixel[1] + darkness_threshold), pixel[2]
    return pixel[0], pixel[1], min(255, pixel[2] + darkness_threshold)


def darken2(pixel, x, y, paramlist):
    smallest = min(pixel[0], pixel[1], pixel[2])
    if smallest == pixel[0]:
        return 0, pixel[1], pixel[2]
    elif smallest == pixel[1]:
        return pixel[0], 0, pixel[2]
    return pixel[0], pixel[1], 0


def darken3(pixel, x, y, paramlist=[]):
    darkness_threshold = paramlist[0] if len(paramlist) > 0 else 20
    smallest = min(pixel[0], pixel[1], pixel[2])
    if smallest == pixel[0]:
        return max(0, pixel[0] - darkness_threshold), pixel[1], pixel[2]
    elif smallest == pixel[1]:
        return pixel[0], max(0, pixel[1] - darkness_threshold), pixel[2]
    return pixel[0], pixel[1], max(0, pixel[2] - darkness_threshold)


def replace_black_with_random(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if not is_above_darkness_threshold(pixel, thresh):
        return random_color()
    return pixel


def replace_black_with_random_dark(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if not is_above_darkness_threshold(pixel, thresh):
        return random_dark_color()
    return pixel


def replace_black_with_random_light(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if not is_above_darkness_threshold(pixel, thresh):
        return random_light_color()
    return pixel


most_recent_nondark = white


def replace_black_with_most_recent_nonblack(pixel, x, y, paramlist=[]):
    global most_recent_nondark
    if len(paramlist) == 0:
        thresh = 386
    else:
        thresh = paramlist[0]

    if not is_above_darkness_threshold(pixel, thresh):
        return most_recent_nondark

    most_recent_nondark = pixel
    return pixel


def replace_black_within_box_with_most_recent_nonblack(pixel, x, y, paramlist=[]):
    global most_recent_nondark
    if len(paramlist) < 5:
        print('error: 5 params required')
        exit(1)
    # if not within box, no-op    
    if x >= paramlist[1] and x <= paramlist[2] and y >= paramlist[3] and y <= paramlist[4]:

        thresh = paramlist[0]

        if not is_above_darkness_threshold(pixel, thresh):
            return most_recent_nondark

    most_recent_nondark = pixel
    return pixel


def grid(pixel, x, y, paramlist=[]):
    if len(paramlist) == 0:
        x_modulus = 50
    else:
        x_modulus = paramlist[0]
    if len(paramlist) < 2:
        y_modulus = 50
    else:
        y_modulus = paramlist[1]

    if x % x_modulus == 0 or y % y_modulus == 0:
        return black
    return pixel


def white_grid(pixel, x, y, paramlist=[]):
    if len(paramlist) == 0:
        x_modulus = 50
    else:
        x_modulus = paramlist[0]
    if len(paramlist) < 2:
        y_modulus = 50
    else:
        y_modulus = paramlist[1]

    if x % x_modulus == 0 or y % y_modulus == 0:
        return white
    return pixel


def replace_white_with_random(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if is_above_darkness_threshold(pixel, thresh):
        return random_color()
    return pixel


def replace_white_with_random_light(pixel, x, y, paramlist=[]):
    thresh = 386 if len(paramlist) == 0 else paramlist[0]
    if is_above_darkness_threshold(pixel, thresh):
        return random_light_color()
    return pixel


def add_noise(pixel, x, y, paramlist=[]):
    darkness_threshold = paramlist[0] if len(paramlist) > 0 else 10
    return noise_or_color(pixel, darkness_threshold)


def add_noise2(pixel, x, y, paramlist=[]):
    darkness_threshold = paramlist[0] if len(paramlist) > 0 else 10
    noise_amount = paramlist[1]
    return noisy_color(pixel, darkness_threshold, noise_amount)


def sketch(pixel, paramlist):
    pass  # 2 DO


def stipple(pixel, x, y, paramlist=[]):
    thresh = paramlist[0] if len(paramlist) > 0 else 200
    test = random.randrange(0, thresh)
    sumchan = sum_channels(pixel)
    return black if sumchan < test else white


def enhanced_stipple(pixel, x, y, paramlist=[]):
    leave_black = 200 if len(paramlist) < 1 else paramlist[0]
    leave_white = 600 if len(paramlist) < 2 else paramlist[1]
    brightness = sum_channels(pixel)
    if brightness > leave_white:
        return white
    if brightness < leave_black:
        return black
    test = random.randrange(leave_black, leave_white)
    return black if brightness < test else white


def tri_tone(pixel, x, y, paramlist=[]):
    threshold1 = paramlist[0] if len(paramlist) > 0 else 200
    threshold2 = paramlist[1] if len(paramlist) > 1 else 100 + threshold1
    sumchan = sum_channels(pixel)
    if sumchan < threshold1:
        return black
    if sumchan < threshold2:
        return 127, 127, 127
    return white


def tetra_tone(pixel, x, y, paramlist=[]):
    threshold1 = paramlist[0] if len(paramlist) > 0 else 100
    threshold2 = paramlist[1] if len(paramlist) > 1 else 100 + threshold1
    threshold3 = paramlist[2] if len(paramlist) > 2 else 100 + threshold2
    sumchan = sum_channels(pixel)
    if sumchan < threshold1:
        return black
    if sumchan < threshold2:
        return 90, 90, 90
    if sumchan < threshold2:
        return 180, 180, 180
    return white


def blue_tone(pixel, x, y, paramlist=[]):
    threshold1 = paramlist[0] if len(paramlist) > 0 else 100
    threshold2 = paramlist[1] if len(paramlist) > 1 else 100 + threshold1
    threshold3 = paramlist[2] if len(paramlist) > 2 else 100 + threshold2
    sumchan = sum_channels(pixel)
    if sumchan < threshold1:
        return 0, 0, 50
    if sumchan < threshold2:
        return 90, 90, 140
    if sumchan < threshold2:
        return 180, 180, 230
    return 230, 230, 255


def whiten(pixel, x, y, paramlist=[]):
    addend = paramlist[0] if len(paramlist) > 0 else 20
    (r, g, b) = pixel
    r = min(r + addend, 255)
    g = min(g + addend, 255)
    b = min(b + addend, 255)
    return r, g, b


def abstract(pixel, x, y, paramlist=[]):
    chance = paramlist[0] if len(paramlist) > 0 else 20
    r = (255 - pixel[0]) if random.randrange(0, 100) < chance else pixel[0]
    g = (255 - pixel[1]) if random.randrange(0, 100) < chance else pixel[1]
    b = (255 - pixel[2]) if random.randrange(0, 100) < chance else pixel[2]
    return r, g, b


def blue_shift(pixel, x, y, paramlist=[]):
    addend = paramlist[0] if len(paramlist) > 0 else 20
    minuend = addend // 2
    (r, g, b) = pixel
    b = min(b + addend, 255)
    g = max(g - minuend, 0)
    r = max(r - minuend, 0)
    return r, g, b


def green_shift(pixel, x, y, paramlist=[]):
    addend = paramlist[0] if len(paramlist) > 0 else 20
    minuend = addend // 2
    (r, g, b) = pixel
    g = min(g + addend, 255)
    b = max(b - minuend, 0)
    r = max(r - minuend, 0)
    return r, g, b


def red_shift(pixel, x, y, paramlist=[]):
    addend = paramlist[0] if len(paramlist) > 0 else 20
    minuend = addend // 2
    (r, g, b) = pixel
    r = min(r + addend, 255)
    g = max(g - minuend, 0)
    b = max(b - minuend, 0)
    return r, g, b


def false_color(pixel, x, y, paramlist=[]):
    black_threshold = paramlist[0] if len(paramlist) > 0 else 20
    white_threshold = paramlist[1] if len(paramlist) > 1 else 200
    value = sum_channels(pixel)
    if value < black_threshold:
        return black
    elif value > white_threshold:
        return white
    else:
        (r, g, b) = pixel
        if (r >= g) and (r >= b):
            return 255, 0, 0
        elif (g >= r) and (g >= b):
            return 0, 255, 0
        else:
            return 0, 0, 255


def autocontrast(img, paramlist=[]):
    cutoff = 10 if len(paramlist) == 0 else paramlist[0]
    return ImageOps.autocontrast(img, cutoff)


def posterize(img, paramlist=[]):
    nbits = 4 if len(paramlist) == 0 else paramlist[0]
    print('nbits =', nbits)
    return ImageOps.posterize(img, nbits)


def posterize2(pixel, x, y, paramlist=[]):
    r = pixel[0] // 30
    g = pixel[1] // 30
    b = pixel[2] // 30
    return r * 30, g * 30, b * 30


def quantize(img, paramlist=[]):
    threshold = 20 if len(paramlist) == 0 else paramlist[0]
    img = img.convert("P", palette=Image.ADAPTIVE, colors=threshold)
    return img.convert("RGB")


def quantize_then_greyscale(img, paramlist=[]):
    threshold = 20 if len(paramlist) == 0 else paramlist[0]
    img = ImageOps.grayscale(img)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=threshold)
    return img.convert("RGB")


def greyscale_then_quantize(img, paramlist=[]):
    threshold = 20 if len(paramlist) == 0 else paramlist[0]
    img = img.convert("P", palette=Image.ADAPTIVE, colors=threshold)
    img = ImageOps.grayscale(img)
    return img.convert("RGB")


# reduce resolution by averaging [step-x-step] squares    
def deres(img_pix, out_img, width, height, paramlist):
    step = 3 if len(paramlist) == 0 else paramlist[0]
    for x in range(0, width, step):
        for y in range(0, height, step):
            sum_r = 0
            sum_g = 0
            sum_b = 0
            npix = 0
            for dx in range(step):
                for dy in range(step):
                    if x + dx < width:
                        if y + dy < height:
                            sum_r += img_pix[x + dx, y + dy][0]
                            sum_g += img_pix[x + dx, y + dy][1]
                            sum_b += img_pix[x + dx, y + dy][2]
                            npix += 1
            # average
            avg_r = sum_r // npix
            avg_g = sum_g // npix
            avg_b = sum_b // npix
            for dx in range(step):
                for dy in range(step):
                    if x + dx < width:
                        if y + dy < height:
                            out_img[x + dx, y + dy] = (avg_r, avg_g, avg_b)


def convolve(img_pix, out_img, width, height, convolution, paramlist):
    max_i = width - 1
    max_j = height - 1
    for j in range(height):
        # print( j, end=',' ) # op ent
        for i in range(width):
            neighbors = []
            if i > 0:
                neighbors.append(img_pix[i - 1, j])  # left
                if j > 0:
                    neighbors.append(img_pix[i - 1, j - 1])  # upper left corner
                if j < max_j:
                    neighbors.append(img_pix[i - 1, j + 1])  # lower left corner
            if i < max_i:
                neighbors.append(img_pix[i + 1, j])  # right
                if j > 0:
                    neighbors.append(img_pix[i + 1, j - 1])  # upper right corner
                if j < max_j:
                    neighbors.append(img_pix[i + 1, j + 1])  # lower right corner
            if j > 0:
                neighbors.append(img_pix[i, j - 1])  # above
            if j < max_j:
                neighbors.append(img_pix[i, j + 1])  # below

            out_img[i, j] = convolution(img_pix[i, j], neighbors, paramlist)


# create layers based on brightness
def layerize(img, im_path, layerfolder, opcode, params):
    num_layers = 10 if len(params) == 0 else params[0]
    composite = False if len(params) < 2 else (str(params[1]) == '1')
    # debug:
    # print( params, num_layers, composite, len(params), params[1], (str(params[1]) == '1') )
    img_pix = img.load()
    (width, height) = img.size
    new_images = []
    for layer in range(num_layers):
        new_images.append(Image.new(img.mode, (width, height)))

    divisor = sum_channels(white) // num_layers
    for i in range(width):
        for j in range(height):
            dest_layer = sum_channels(img_pix[i, j]) // divisor
            for layer in range(num_layers):
                dest_pix = new_images[layer].load()
                test = (layer == dest_layer) if not composite else (layer <= dest_layer)
                if test:
                    dest_pix[i, j] = img_pix[i, j]
                else:
                    dest_pix[i, j] = white

    for layer in range(num_layers):
        layernum = [str(layer)]
        save(new_images[layer], im_path, layerfolder, opcode, layernum)


def process_with_paramlist(img_pix, out_img, width, height, pixelop, paramlist):
    for y in range(height):
        for x in range(width):
            out_img[x, y] = pixelop(img_pix[x, y], x, y, paramlist)


def process_with_loop_inside_pixelop(img_pix, out_img, width, height, pixelop, paramlist):
    pixelop(img_pix, out_img, width, height, paramlist)


def save(new_image, in_path, out_path, opcode, paramlist=[]):
    sep_char = '\\' if '\\' in in_path else '/'
    try:
        left_part = in_path.split(sep_char)[-1]
        left_part = left_part[:-4]
        newfile = out_path + '/' + left_part + '-' + opcode
        for p in paramlist:
            newfile += '-' + str(p)
        newfile += '.jpg'
        print('writing to ' + newfile)
        new_image.save(newfile)
        return newfile
    except:
        print('writing to default.jpg')
        new_image.save('default.jpg')
        return 'default.jpg'


# if folder doesn't exist, create it
# NOTE: if a file named 'foldername' exists, this doesn't work
def ensurefolder(foldername):
    try:
        check = os.stat(foldername)
    except:
        os.mkdir(foldername)


def printfilters(filters):
    for f in sorted(filters):
        print(f, filters[f])


def printops(ops):
    for o in sorted(ops):
        print(o, ops[o])


def load_filters():
    filters = {}
    filters['B'] = ImageFilter.BLUR
    filters['C'] = ImageFilter.CONTOUR
    filters['D'] = ImageFilter.DETAIL
    filters['EE'] = ImageFilter.EDGE_ENHANCE
    filters['EEM'] = ImageFilter.EDGE_ENHANCE_MORE
    filters['EM'] = ImageFilter.EMBOSS
    filters['FE'] = ImageFilter.FIND_EDGES
    filters['ME'] = ImageFilter.MedianFilter
    filters['MN'] = ImageFilter.MinFilter
    filters['MO'] = ImageFilter.ModeFilter
    filters['MX'] = ImageFilter.MaxFilter
    filters['S'] = ImageFilter.SMOOTH
    filters['SH'] = ImageFilter.SHARPEN
    filters['SM'] = ImageFilter.SMOOTH_MORE
    return filters


def load_img_ops():
    img_ops = {}
    img_ops['a'] = autocontrast
    img_ops['EQ'] = ImageOps.equalize
    img_ops['gq'] = greyscale_then_quantize  # difference between these 2 ...
    img_ops['GS'] = ImageOps.grayscale
    img_ops['IV'] = ImageOps.invert
    img_ops['p'] = posterize
    img_ops['q'] = quantize
    img_ops['qg'] = quantize_then_greyscale  # ... is usually pretty subtle
    return img_ops


def load_cp_ops():
    conv_ops = {}
    proc_ops = {}
    proc_ops['a'] = abstract
    proc_ops['ax'] = auto_crosshatch
    proc_ops['b'] = blue_shift
    proc_ops['bbt'] = blacken_below_threshold
    proc_ops['bnw'] = blacken_nonwhite
    proc_ops['bt'] = blue_tint
    proc_ops['bw'] = black_or_white
    conv_ops['c'] = edge_convolve
    conv_ops['c2'] = edge_convolve2
    conv_ops['c3'] = edge_convolve3
    conv_ops['c4'] = edge_convolve4
    conv_ops['c5'] = edge_convolve5
    conv_ops['cm'] = contrast_map
    proc_ops['d'] = darken
    proc_ops['d2'] = darken2
    proc_ops['d3'] = darken3
    proc_ops['di'] = dither
    proc_ops['dr'] = deres
    proc_ops['e'] = enrich
    proc_ops['e2'] = enrich2
    proc_ops['es'] = enhanced_stipple
    proc_ops['f'] = false_color
    proc_ops['g'] = green_shift
    proc_ops['gr'] = grid
    proc_ops['l'] = lighten
    proc_ops['lbt'] = light_blue_tint
    conv_ops['m'] = maybe_black_dot
    proc_ops['n'] = add_noise
    proc_ops['n2'] = add_noise2
    proc_ops['neg'] = negate
    proc_ops['p2'] = posterize2
    proc_ops['px'] = pseudo_crosshatch
    proc_ops['rb'] = replace_black_with_random
    proc_ops['rbd'] = replace_black_with_random_dark
    proc_ops['rbl'] = replace_black_with_random_light
    proc_ops['rbm'] = replace_black_with_most_recent_nonblack
    proc_ops['rbx'] = replace_black_within_box_with_most_recent_nonblack
    conv_ops['rlb'] = remove_lone_black_pixels
    proc_ops['rw'] = replace_white_with_random
    proc_ops['rwl'] = replace_white_with_random_light
    proc_ops['r'] = red_shift
    conv_ops['s'] = scramble
    proc_ops['st'] = stipple
    conv_ops['tew'] = random_color_edge_convolve
    proc_ops['t3'] = tri_tone
    proc_ops['t4'] = tetra_tone
    proc_ops['b4'] = blue_tone
    proc_ops['w'] = whiten
    proc_ops['wg'] = white_grid
    return conv_ops, proc_ops


def load_layer_ops():
    layer_ops = {}
    layer_ops['la'] = layerize
    return layer_ops


if __name__ == '__main__':

    argno = 1
    try:
        print('input file: ' + sys.argv[argno])
        img = Image.open(sys.argv[argno]).convert('RGB')
    except:
        print('Image.open(...) failed!')
        # 2DO: print stack trace, exception msg
        exit(1)

    imgname = sys.argv[argno]

    argno += 1
    try:
        opcode = sys.argv[argno]
    except:
        opcode = 'bw'

    more = True
    argno += 1
    params = []
    while more:
        try:
            params.append(int(sys.argv[argno]))
            argno += 1
        except:
            more = False

    # special cases
    # ----------------
    if opcode == 'h':
        print(img.histogram())
        exit(0)

    if opcode == 'gc':
        threshold = 20000 if len(params) == 0 else params[0]
        print(img.getcolors(threshold))
        exit(0)
    # ----------------        

    filters = load_filters()
    img_ops = load_img_ops()
    conv_ops, proc_ops = load_cp_ops()
    layer_ops = load_layer_ops()

    outfolder = 'newimages'
    layerfolder = 'layers'
    ensurefolder(outfolder)
    ensurefolder(layerfolder)

    outfile = ''

    if opcode in filters:

        new_image = img.filter(filters[opcode])
        outfile = save(new_image, imgname, outfolder, opcode)

    elif opcode in img_ops:

        if len(params) > 0:
            new_image = img_ops[opcode](img, params)
        else:
            new_image = img_ops[opcode](img)
        outfile = save(new_image, imgname, outfolder, opcode, params)

    elif opcode in conv_ops or opcode in proc_ops:

        img_pix = img.load()
        (width, height) = img.size

        new_image = Image.new(img.mode, (width, height))

        if opcode in conv_ops:
            convolve(img_pix, new_image.load(), width, height, conv_ops[opcode], params)
        elif opcode == 'dr':  # another special case
            process_with_loop_inside_pixelop(img_pix, new_image.load(), width, height, proc_ops[opcode], params)
        else:
            process_with_paramlist(img_pix, new_image.load(), width, height, proc_ops[opcode], params)

        outfile = save(new_image, imgname, outfolder, opcode, params)

    elif opcode in layer_ops:

        layer_ops[opcode](img, imgname, layerfolder, opcode, params)

    else:
        print('error: 3rd arg [', opcode, '] unrecognized')
        print('\nrecognized filter codes:\n')
        printfilters(filters)
        print('\nrecognized "convolve op" codes:\n')
        printops(conv_ops)
        print('\nrecognized "process op" codes:\n')
        printops(proc_ops)
        print()
        # pursuant to cm: contrast_map()
        maxed = 8 * euclidean_dist((0, 0, 0), (255, 255, 255))
        print(maxed, maxed / 256, 256 * (maxed / 256), maxed / 6100)
        exit(1)

    if outfile != '' and 'pos' not in os.name:
        image = cv2.imread(outfile)
        view_image(outfile)
    else:
        print('sorry, view_image ndg on posix')
