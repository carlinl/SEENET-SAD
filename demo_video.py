import argparse
import cv2
import torch
import numpy as np

from model import SCNN
from model_ENET_SAD import ENet_SAD
from model_SEENET_SAD import SEENet_SAD
from utils.prob2lines import getLane

from utils.prob2lines.lane_detection_funtions import roneld_lane_detection
from utils.transforms import *

import time
from multiprocessing import Process, JoinableQueue, SimpleQueue
from threading import Lock

img_size = (800, 288)
#net = SCNN(input_size=(800, 288), pretrained=False)
net = ENet_SAD((800,288), sad=False)
seenet = SEENet_SAD((800, 288), sad=False)

# CULane mean, std
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
# mean=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)
transform_img = Resize(img_size)
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))

pipeline = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", '-i', type=str, default="experiments/Zhejiang_cloud_crude_20201103175630_cruve.mp4", help="Path to demo video")
    parser.add_argument("--weight_path", '-w', type=str, default="experiments/exp1/exp1_best.pth", help="Path to model weights")
    parser.add_argument("--weight_path1", '-x', type=str, default="experiments/exp2/exp1_best.pth", help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args

def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv2.Canny(blur, 50, 150)

    return canny

def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(150, 200), (580, 200), (350, 100)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    print("lallalalalal",lines)
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        print(x1, y1, x2, y2 )
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 50)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize


# The following frees up resources and closes all windows


def network(net, img):
    seg_pred, exist_pred = net(img)[:2]
    seg_pred = seg_pred.detach().cpu()
    exist_pred = exist_pred.detach().cpu()
    return seg_pred, exist_pred

def visualize(img, seg_pred, exist_pred):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            lane_img[coord_mask == (i + 1)] = color[i]
    # points_list = [(150, 200), (580, 200), (350, 100)]
    #
    # for point in points_list:
    #     cv2.circle(img, point, 40, (0,0,255))

    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    return img


def pre_processor(arg):
    img_queue, video_path = arg
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        if img_queue.empty():
            ret, frame = cap.read()
            if ret:
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = transform_img({'img': frame})['img']
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = transform_to_net({'img': img})['img']
                x.unsqueeze_(0)

                img_queue.put(x)
                img_queue.join()
            else:
                break

def post_processor(arg):
    img_queue, arg_visualize = arg

    while True:
        if not img_queue.empty():
            x, seg_pred, exist_pred = img_queue.get()
            seg_pred = seg_pred.numpy()[0]
            exist_pred = exist_pred.numpy()

            exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

            print(exist)
            for i in getLane.prob2lines_CULane(seg_pred, exist):
                print(i)

            if arg_visualize:
                frame = x.squeeze().permute(1, 2, 0).numpy()
                img = visualize(frame, seg_pred, exist_pred)
                cv2.imshow('input_video', frame)
                cv2.imshow("output_video", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            pass

def main():
    args = parse_args()
    video_path = args.video_path
    weight_path = args.weight_path
    weight_path1 = args.weight_path1


    if pipeline:
        input_queue = JoinableQueue()
        pre_process = Process(target=pre_processor, args=((input_queue, video_path),))
        pre_process.start()

        output_queue = SimpleQueue()
        post_process = Process(target=post_processor, args=((output_queue, args.visualize),))
        post_process.start()
    else:
        cap = cv2.VideoCapture(video_path)

    save_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(save_dict['net'])
    net.eval()

    save_dict1 = torch.load(weight_path1, map_location='cpu')
    seenet.load_state_dict(save_dict1['net'])
    seenet.eval()
    #net.cuda()

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output_cv.mp4', fourcc, 20.0, (800, 288))
    out1 = cv2.VideoWriter('output_cv11.mp4', fourcc, 20.0, (800, 288))
    while True:
        if pipeline:
            loop_start = time.time()
            x = input_queue.get()
            input_queue.task_done()

            gpu_start = time.time()
            seg_pred, exist_pred = network(net, x)
            gpu_end = time.time()

            output_queue.put((x, seg_pred, exist_pred))

            loop_end = time.time()

        else:
            if not cap.isOpened():
                break

            ret, frame = cap.read()

            if ret:
                loop_start = time.time()
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = transform_img({'img': frame})['img']
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #img_black = np.zeros([288, 800, 3], np.uint8)

                x = transform_to_net({'img': img})['img']

                x.unsqueeze_(0)

                gpu_start = time.time()
                seg_pred, exist_pred = network(net, x)
                gpu_end = time.time()

                start = time.time()
                seg1_pred, exist1_pred = network(seenet, x)
                end = time.time()

                seg_pred = seg_pred.numpy()[0]
                exist_pred = exist_pred.numpy()

                seg1_pred = seg1_pred.numpy()[0]
                exist1_pred = exist1_pred.numpy()


                exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

                print('enet lane exist: ', exist)

                exist1 = [1 if exist1_pred[0, i] > 0.5 else 0 for i in range(4)]

                print('se-enet lane exist: ', exist1)



                for i in getLane.prob2lines_CULane(seg_pred, exist):
                    print(i)


                for i in getLane.prob2lines_CULane(seg1_pred, exist1):
                    print(i)



                loop_end = time.time()

                if args.visualize:
                    img = visualize(img, seg_pred, exist_pred)
                    img1 = visualize(img, seg1_pred, exist1_pred)
                    cv2.imshow('enet_output_video', img)
                    cv2.imshow("seenet_output_video", img1)

                    # write the flipped frame
                    out.write(img)
                    out1.write(img1)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        print("enent_runtime: ", gpu_end - gpu_start, "se_enet runtime; ", (end - start))
        print("total_runtime:", loop_end - loop_start, "FPS:", int(1 / (loop_end - loop_start)))

    cap.release()
    out.release()
    out1.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
