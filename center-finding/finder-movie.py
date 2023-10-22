import argparse
import numpy as np
import cv2


def nothing(x):
    pass


def main(args):
    vid = cv2.VideoCapture(args.input_video)
    if vid is None:
        print(f'Unable to open video at {args.input_video}')
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    size = (frame_width, frame_height)
    frame_grabbed, frame = vid.read()

    result = cv2.VideoWriter('filename.avi',
                             cv2.VideoWriter_fourcc(*'XVID'),
                             60, size, 0)
    result2 = cv2.VideoWriter('filename1.avi',
                              cv2.VideoWriter_fourcc(*'XVID'),
                              60, size, 0)
    cv2.namedWindow("Obraz przeksztalcony", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("1.LowH", "Obraz przeksztalcony", -0, 640, nothing)

    while frame_grabbed:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([160, 140, 50])
        upper_red = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14)))

        oMoments = cv2.moments(mask)
        dM01 = oMoments['m01']
        dM10 = oMoments['m10']
        dArea = oMoments['m00']
        x = 0
        y = 0
        if dArea != 0:
            x = int(dM10 / dArea)
            y = int(dM01 / dArea)
        dist = abs(320 - x)

        cv2.setTrackbarPos("1.LowH", "Obraz przeksztalcony", dist)

        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(frame, "centroid", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Centroid", frame)

        result.write(frame)
        result2.write(mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_grabbed, frame = vid.read()
    vid.release()
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description=('This script divides video'
                                                  'stream into R,G,B channels'))
    parser.add_argument('-i',
                        '--input_video',
                        type=str,
                        required=True,
                        help='A video file that will be processed.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
