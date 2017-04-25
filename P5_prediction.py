import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from my_cv_tools import get_hog_features
from my_cv_tools import bin_spatial
from my_cv_tools import color_hist
from my_cv_tools import convert_color
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# from my_cv_tools import single_img_features
# from lesson_functions import *

# training parameter
dist_pickle = pickle.load(open("svc_pickle.pkl", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
print("svc spac :", svc.best_estimator_,
      svc.best_params_, svc.best_score_)
print("orient :", orient)
print("pix_per_cell :", pix_per_cell)
print("cell_per_block :", cell_per_block)
print("spatial_size :", spatial_size)
print("hist_bins : ", hist_bins)


class CarBox():
    """
    @desc store the bbox in time-series
    """
    def __init__(self, num):
        # past N data of detected boxes
        self.box_sequence = []

        # the number of data to keep in the past
        self.N = num

    def update_recent_box(self, boxes):
        if len(self.box_sequence) > self.N:
            self.box_sequence.pop(0)

        self.box_sequence.append(boxes)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    # Return the image
    return img


def find_cars(img, ystart, ystop, cspace, scale, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, hog_channel,
              spatial_size, hist_bins):
    """
    @desc : 1. extract features using hog sub-sampling
            2. feature extraction
            3. prediction based on the trained model
            4. return detected box list
    @return : box list
    @param ystart : upper bound of search area
    @param ystop : lower bound of search area
    @param scale : if big, try to find close car,
                   if small, try to find far car
    """
    box_list = []
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255  # can be removed

    # crop image
    img_tosearch = img[ystart:ystop, :, :]
    color_param = "RGB2" + cspace
    ctrans_tosearch = convert_color(img_tosearch,
                                    conv=color_param)

    # tweak window size depending on how far you look at
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/scale),
                                      np.int(imshape[0]/scale)))
        img_tosearch = cv2.resize(img_tosearch,
                                  (np.int(imshape[1]/scale),
                                   np.int(imshape[0]/scale)))

    # each channel
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    # nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute each channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient,
                            pix_per_cell, cell_per_block,
                            feature_vec=False)
    hog2 = get_hog_features(ch2, orient,
                            pix_per_cell, cell_per_block,
                            feature_vec=False)
    hog3 = get_hog_features(ch3, orient,
                            pix_per_cell, cell_per_block,
                            feature_vec=False)

    for i, xb in enumerate(range(nxsteps)):
        for yb in range(nysteps):
            # Define patch (starting position)
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Feature 1 : HOG (sub sampling for this patch)
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            if hog_channel is 0:
                hog_features = hog_feat1
            elif hog_channel is 1:
                hog_features = hog_feat2
            elif hog_channel is 2:
                hog_features = hog_feat3
            elif hog_channel == 'ALL':
                hog_features = np.hstack((hog_feat1,
                                          hog_feat2,
                                          hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[
                ytop:ytop+window, xleft:xleft+window],
                                (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg,
                                           size=spatial_size)
            hist_features = color_hist(subimg,
                                       nbins=hist_bins)

            # Scale features and make a prediction
            X = np.hstack((spatial_features, hist_features,
                           hog_features)).reshape(1, -1)

#             # DEBUG : feature from single_img_features
#             X = single_img_features(img_tosearch[
#                 ytop:ytop+window, xleft:xleft+window],
#                                     color_space="YCrCb",
#                                     hog_channel='ALL'
#             ).reshape(1, -1)

            test_features = X_scaler.transform(X)
            test_prediction = svc.predict(test_features)

            # Draw Rectangle
            if test_prediction == 1:  # if result is car
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append(((xbox_left,
                                 ytop_draw+ystart),
                                (xbox_left+win_draw,
                                 ytop_draw+win_draw+ystart)))

#                cv2.rectangle(draw_img,
#                              (xbox_left, ytop_draw+ystart),
#                              (xbox_left+win_draw,
#                               ytop_draw+win_draw+ystart),
#                              (0, 0, 255), 6)
#    plt.imshow(draw_img)
#    plt.show()
    
    return box_list


def finalize_cars(img, box_list, thresh):
    """
    @desc : 1. heatmap by overlaying the detected box
            2. apply threshold to remove noise
            3. prediction based on the trained model
            4. extrat labeled area
    @return : labeled image
    @param boxlist : detected car (box) boundary infor
    """

    # add heat map
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, thresh)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img


def pipeline(img):
    """
    @desc : return rectangular
    @return : final conclusion of car location
    @param img : the most recent img
    """
    global svc
    global X_scaler
    global orient
    global pix_per_cell
    global cell_per_block
    global spatial_size
    global hist_bins

    # fixed parameter
    cspace = "YCrCb"
    hog_channel = 0

    # search paramter
    ystarts = [400, 400, 400, 400]
    ystops = [500, 560, 620, 670]
    scales = [1.0, 1.5, 2.0, 2.5]

    # add the detected box
    box_lists = []
    for ystart, ystop, scale in zip(ystarts, ystops, scales):
        box_list = find_cars(img,  # image file
                             ystart,  # target y area start
                             ystop,  # target y area end
                             cspace,   # color space such as 'YCrCb'
                             scale,  # scale demined by distance to car
                             svc,  # classifier
                             X_scaler,  # scaler for normalization
                             orient,  # number of direction feature
                             pix_per_cell,  # pixel per cell (8)
                             cell_per_block,  # cell per block (2)
                             hog_channel,  # target hog channel
                             spatial_size,  # target resize feature
                             hist_bins)  # target hist
        box_lists += box_list

    # overlay multiple boxes
    thresh = 2
    out_image = finalize_cars(img, box_lists, thresh)

    return out_image


if __name__ == '__main__':
    # set mode
    exp_mode = 'video'

    # image mode
    if exp_mode == 'image':
        # read data
        img = mpimg.imread('test_images/test5.jpg')
        out_images = pipeline(img)
        plt.imshow(out_images)
        plt.show()

    # video mode
    if exp_mode == 'video':
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(pipeline)

        # output
        white_output = 'project_video_out.mp4'
        white_clip.write_videofile(white_output, audio=False)
