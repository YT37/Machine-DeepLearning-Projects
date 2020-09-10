from cv2 import (
    COLOR_BGR2GRAY,
    CascadeClassifier,
    VideoCapture,
    cvtColor,
    destroyAllWindows,
    imshow,
    rectangle,
    waitKey,
)~

face = CascadeClassifier("Face.xml")
eye = CascadeClassifier("Eye.xml")
smile = CascadeClassifier("Smile.xml")


def detect(gray, frame):
    for x, y, w, h in face.detectMultiScale(gray, 1.3, 5):
        rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roiGray = gray[y : y + h, x : x + w]
        roiColor = frame[y : y + h, x : x + w]

        for ex, ey, ew, eh in eye.detectMultiScale(roiGray, 1.1, 22):
            rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        for sx, sy, sw, sh in smile.detectMultiScale(roiGray, 1.7, 22):
            rectangle(roiColor, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    return frame


capture = VideoCapture(0)

while True:
    _, frame = capture.read()
    canvas = detect(cvtColor(frame, COLOR_BGR2GRAY), frame)
    imshow("Video", canvas)

    if waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
destroyAllWindows()
