import cv2
import numpy as np
import os

def initialize_background_subtractor():
    return cv2.createBackgroundSubtractorKNN( dist2Threshold=400,detectShadows=True) # ! for the first and second video.
    # return cv2.createBackgroundSubtractorKNN( dist2Threshold=1000,detectShadows=True) # !for the fourth video.
    # return cv2.createBackgroundSubtractorMOG2( varThreshold=50,detectShadows=True)


def apply_mask(frame, background_object, kernel):
    fgmask = background_object.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    # fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    # fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    return fgmask

def find_and_draw_cars(frame, fgmask, roi_polygon, output_folder):
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_copy = frame.copy()
    unique_image_id = 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            x, y, width, height = cv2.boundingRect(cnt)
            
            if cv2.pointPolygonTest(roi_polygon, (x + width // 2, y + height // 2), False) >= 0:
                cv2.rectangle(frame_copy, (x, y), (x + width, y + height), (0, 0, 255), 2)
                cv2.putText(frame_copy, 'Violation detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                car_region = frame[y:y + height, x:x + width]
                unique_image_id += 1
                image_name = os.path.join(output_folder, f"car_{unique_image_id}.jpg")
                cv2.imwrite(image_name, car_region)
            else:
                cv2.rectangle(frame_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return frame_copy

def main():
    roi_polygon = np.array([
        # [(924, 366), (975, 366), (1000, 736),(927,736)]    #!for the video 1_1.
        [(595, 241), (652, 241), (689, 570), (606, 570)]   #! for the first video.
        # [(550, 307), (620, 307), (914, 568), (794, 568)] #! for the second video.
        #   [(644, 98), (673, 98), (698, 585),(608,585)]  #!for fourth video.
        # [(746, 403), (757, 403), (960, 565), (880, 565)]  #!for the fifth video.
    
    ])

    output_folder = "D:\\3rd_year\\1st_sem\CS_314_FINAL_PROJECT\\video_1_detected_cars"
    # output_folder = "D:\\3rd_year\\1st_sem\CS_314_FINAL_PROJECT\\video_2_detected_cars" 
    # output_folder = "D:\\3rd_year\\1st_sem\CS_314_FINAL_PROJECT\\video_3_detected_cars"
    # output_folder = "D:\\3rd_year\\1st_sem\CS_314_FINAL_PROJECT\\video_4_detected_cars"
    # output_folder = "D:\\3rd_year\\1st_sem\CS_314_FINAL_PROJECT\\video_5_detected_cars"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # video = cv2.VideoCapture("dataset\\video_1_1")
    video = cv2.VideoCapture("dataset\\video_1.mp4")
    # video = cv2.VideoCapture("dataset\\video_2.mp4")
    # video = cv2.VideoCapture("dataset\\video_4.mp4")
    # video = cv2.VideoCapture("dataset\\video_5.mp4")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    background_object = initialize_background_subtractor()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        fgmask = apply_mask(frame, background_object, kernel)
        frame_copy = find_and_draw_cars(frame, fgmask, roi_polygon, output_folder)

        foreground_part = cv2.bitwise_and(frame, frame, mask=fgmask)
        fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        stacked = np.hstack((frame, foreground_part, frame_copy))

        cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked, None, fx=0.2, fy=0.6))
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
