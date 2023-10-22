import argparse
import cv2
import numpy as np


def nothing(x):
    pass


def main(args):
    img = cv2.imread(args.input_image)
    if img is None:
        print(f'Unable to load image at {args.input_image}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 220, 3)
    output = img.copy()
    # wykrywanie kół
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 40, param1=9,
                               param2=27, minRadius=20, maxRadius=80)
    # wykrywanie linii
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 200, minLineLength=150, maxLineGap=250)
    print(lines)
    count_grosze = 0
    count_zlote = 0
    coin_area = 0
    avg_zl_area = 0
    tray_grosze = 0
    tray_zlote = 0
    # szukanie koordynatow tacy
    if lines is not None:
        max_x, max_y, min_x, min_y = 0, 0, 9999, 9999
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if max(x1, x2) > max_x:
                max_x = max(x1, x2)
            if max(y1, y2) > max_y:
                max_y = max(y1, y2)
            if min(x1, x2) < min_x:
                min_x = min(x1, x2)
            if min(y1, y2) < min_y:
                min_y = min(y1, y2)
        tray_area = (max_x - min_x) * (max_y - min_y)
    if circles is not None:
        # zmiana koordynatów i promienia na int
        circles = np.round(circles[0, :]).astype("int")
        cv2.line(output, (min_x, min_y), (min_x, max_y), (0, 255, 0), 2)
        cv2.line(output, (min_x, max_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.line(output, (max_x, max_y), (max_x, min_y), (0, 255, 0), 2)
        cv2.line(output, (min_x, min_y), (max_x, min_y), (0, 255, 0), 2)
        for (x, y, r) in circles:
            # zaznaczanie koła i środka
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            coin_area += 3.14159 * r * r
            if r < 34.1:
                count_grosze += 1
                cv2.putText(output, 'gr', org=(x, y), fontFace=2, fontScale=1, color=(255, 0, 0))
                if max_x > x > min_x and max_y > y > min_y:
                    tray_grosze += 5
            else:
                count_zlote += 1
                avg_zl_area += 3.14159 * r * r
                cv2.putText(output, 'zl', org=(x, y), fontFace=2, fontScale=1, color=(255, 0, 0))
                if max_x > x > min_x and max_y > y > min_y:
                    tray_zlote += 5
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        avg_zl_area /= count_zlote
        cv2.imshow("output", canny)
        cv2.waitKey(0)
        cv2.imshow("output", np.hstack([img, output]))
        print("Grosze: ", count_grosze, " Zlote: ", count_zlote, "\nTotal coin area: ", coin_area)
        print("Tray area: ", tray_area)
        print("5 zl area is ", round(tray_area/avg_zl_area), " times smaller than tray area")
        print("Total of tray coins: ", tray_zlote, " zl, ", tray_grosze, " gr")
        cv2.waitKey(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description=('This script marks all of '
                                                  'the circles in the image'))
    parser.add_argument('-i',
                        '--input_image',
                        type=str,
                        required=True,
                        help='An image file that will be processed.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
