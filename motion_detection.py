# import necessary packages
import imutils
from imutils.video import VideoStream
import time
import cv2

# start video
vs = VideoStream(src=0).start()
time.sleep(2.0)

bg_subtract = cv2.createBackgroundSubtractorMOG2(history=250)

# loop over all frames until S key is pressed
while True:
    # get the frame
    frame = vs.read()

    # if we can't get the frame, end the video stream
    if frame is None:
        break

    # resize the frame
    frame = imutils.resize(frame, width=500)

    fg_mask = bg_subtract.apply(frame)
    cv2.imshow("1. Foreground", fg_mask)

    # apply Gaussian blur
    frame_gray = cv2.GaussianBlur(fg_mask, (21, 21), 0)
    cv2.imshow("2. Gaussian Blur", frame_gray)

    # treshold frame difference  to filter the regions with significant change
    # if delta < 25 we discard the pixel, else we set it to white
    frame_thresh = cv2.threshold(frame_gray, 25, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("3. Threshold", frame_thresh)

    # dilate the threshold image
    frame_thresh = cv2.dilate(frame_thresh, None, iterations=2)
    cv2.imshow("4. Threshold dilated", frame_thresh)

    # apply contour detection to find the outlines of the white regions
    contours = cv2.findContours(frame_thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # loop over the contours in order to filter the irrelevant contours
    for contour in contours:
        if cv2.contourArea(contour) < 15000:
            continue
        # if the contour is larger than the minimum supplied area,
        # compute the bounding box for the moving object
        (x_left_top, y_left_top, width, height) = cv2.boundingRect(contour)
        x_right_top = x_left_top + width
        y_left_bottom = y_left_top + height
        cv2.rectangle(frame, (x_left_top, y_left_top), (x_right_top, y_left_bottom),
                      (232, 115, 37), 2)

        # write the position of object in the frame
        label = "left top: (" + str(x_left_top) + ", " + str(y_left_top) + "), right bottom : (" + str(
            x_right_top) + ", " + str(y_left_bottom) + ")"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_left_bottom = max(y_left_bottom, label_size[1])

        cv2.rectangle(frame, (x_left_top, y_left_top - label_size[1]),
                      (x_left_top + label_size[0], y_left_top + base_line),
                      (209, 237, 237), cv2.FILLED)
        cv2.putText(frame, label, (x_left_top, y_left_top), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0))


    cv2.imshow("Motion detection", frame)

    # if user presses the 'q' key, stop the videostream
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        break

# cleanup
vs.stop()
cv2.destroyAllWindows()


