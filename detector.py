import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def do_camera_view(frame):
    height = frame.shape[0]  
    width =  frame.shape[1]  
    src = np.float32([[0, height], [905, height], [0, 0], [width, 0]])
    dst = np.float32([[569, height], [711, height], [0, 0], [width, 0]])
    M = cv.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv.getPerspectiveTransform(dst, src) # Inverse transformation
   
    img = frame[450:(450+height), 0:width] # Apply np slicing for ROI crop
    warped_img = cv.warpPerspective(img, M, (width, height)) # Image warping
    snip = warped_img[0:650,0:width]
    img_inv = cv.warpPerspective(snip, Minv, (width, height)) # Inverse transformation
    snip_inv = img_inv[0:250,0:width]
    camera_view = cv.cvtColor(snip_inv, cv.COLOR_BGR2RGB)
    return camera_view

def do_identify_edges(frame):     
    rgb_planes = cv.split(frame)
    result_planes = []
    result_norm_planes = [] 
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8)) 
        bg_img = cv.medianBlur(dilated_img, 29)
        diff_img = 255 - cv.absdiff(plane, bg_img)   
        norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)  
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)  
        
      
    result_norm = cv.merge(result_norm_planes) 
    #cv.imshow("result_norm", result_norm) 
    gray = cv.cvtColor(result_norm, cv.COLOR_RGB2GRAY)   
    #cv.imshow("gray", gray) 
    gray_blur = cv.GaussianBlur(gray, (11,11), 0)
    #cv.imshow("gray_blur", gray_blur) 
    canny = cv.Canny(gray_blur, 50, 150, 3) 
    #cv.imshow("canny", canny) 

    return canny

def region_of_interest(frame): 
    # create mask to select region of interest
    height = frame.shape[0]  
    width =  frame.shape[1]    
    mask = np.zeros((height, width), dtype="uint8")
    pts = np.array([[180, height], [(width/2.40), 120], [(width/1.7), 120], [(width - 320), height]], dtype=np.int32)
    cv.fillConvexPoly(mask, pts, 255) 
    
    masked = cv.bitwise_and(frame, frame, mask=mask)
    return masked

def do_hough(frame): 
    hough = cv.HoughLinesP(frame, 2, np.pi / 180, 15, np.array([]), minLineLength = 0, maxLineGap = 20)  
    return hough

def draw_lane_lines_v1(frame, lines): 
    line_frame = np.zeros(
        (
            frame.shape[0],
            frame.shape[1],
            3
        ),
        dtype=np.uint8
    )
    frame = np.copy(frame)
    
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:  
            cv.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)   
    return line_frame 

def draw_lane_lines_v2(frame, lines): 
    line_image = np.zeros_like(frame)
    for line in lines:
        if line is not None:
            cv.line(line_image, *line,  [255, 0, 0], 12)
    return cv.addWeighted(frame, 1.0, line_image, 1.0, 0.0)
 
def pixel_points(y1, y2, line): 
    if line is None:
        return None
    if y1 == y2:
        return
    slope, intercept = line
    if slope == 0:
        return 
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(frame, lines): 
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = frame.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def average_slope_intercept(lines): 
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

if __name__== "__main__":
    # The video feed is read in as a VideoCapture object 
    cap = cv.VideoCapture("test_video.hevc") 
    while (cap.isOpened()): 
        ret, frame = cap.read()

        camera_view = do_camera_view(frame)
        #cv.imshow("camera view", camera_view)    

        # identify edges 
        edges = do_identify_edges(camera_view) 
        #cv.imshow("edges", edges)  

        # select region of interest
        masked = region_of_interest(edges)
        #cv.imshow("masked", masked) 

        # identify lane lines
        hough = do_hough(masked) 
        line_img = np.zeros((*masked.shape, 3), dtype=np.uint8)
 
        # Lane Lunes v1
        lines_v1 = draw_lane_lines_v1(line_img, hough)   
        if lines_v1 is not None: 
            output = cv.addWeighted(camera_view, 0.8, lines_v1, 1., 0.)
            #cv.imshow("output - lines v1", output)  

        # Lane Lunes v2
        lines_v2 = draw_lane_lines_v2(line_img, lane_lines(line_img, hough))
        if lines_v2 is not None: 
            output2 = cv.addWeighted(camera_view, 0.8, lines_v2, 1., 0.)
            #cv.imshow("output - lines v2", output2)  
         
            combined_outputs = cv.addWeighted(output, 0.8, lines_v2, 1., 0.)
            cv.imshow("output", combined_outputs)  

        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()