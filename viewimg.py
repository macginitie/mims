import sys
import cv2

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':

    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
    else:
        image = cv2.imread("newimages/bpt-st-600.jpg")
    viewImage(image, 'huh?')