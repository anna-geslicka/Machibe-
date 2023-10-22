import argparse

import cv2
import numpy as np


def nothing(x):
    pass


def main(args):
    img = cv2.imread(args.input_image)
    if img is None:
        print(f'Unable to load image at {args.input_image}')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14)))

    edges = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)


    cv2.imshow("Ball Tracking", edges)
    cv2.waitKey(0)

    # mom = cv2.moments(mask)
    # cx = int(mom["m10"] / mom["m00"])
    # cy = int(mom["m01"] / mom["m00"])
    # cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    #
    # cv2.imshow("Ball Tracking", img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description=('Script for finding center of a circle.'))
    parser.add_argument('-i',
                        '--input_image',
                        type=str,
                        required=True,
                        help='A image file that will be processed.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
