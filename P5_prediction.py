import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import get_hog_features
from lesson_functions import bin_spatial
from lesson_functions import color_hist
from lesson_functions import convert_color
from lesson_functions import single_img_features
from lesson_functions import *


def find_cars(img, ystart, ystop, cspace, scale, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, hog_channel,
              spatial_size, hist_bins):
    """
    @param ystart : upper bound of search area
    @param ystop : lower bound of search area
    @param scale : if big, try to find close car,
                   if small, try to find far car
    @desc : 1. extract features using hog sub-sampling
            2. feature extraction
            3. prediction based on the trained model
            4. draw rectangle
    """
    draw_img = np.copy(img)
#    img = img.astype(np.float32) / 255

    # crop image
    img_tosearch = img[ystart:ystop, :, :]
    color_param = "RGB2" + cspace
    print(color_param)
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

#            print(subimg.shape)
#            plt.imshow(subimg)
#            plt.show()

            # when scale = 1.5
#            print(i)
#            if i == 35:
#                print(ytop, ytop+window)
#                print(xleft, xleft+window)
#                test = img_tosearch[ytop:ytop+window, xleft:xleft+window]
#                plt.imshow(test)
#                plt.show()
#                print(subimg.shape)
#                plt.imshow(subimg)
#                plt.show()

            # when scale = 3.0
#            if i == 21:
#                print(subimg.shape)
#                plt.imshow(subimg)
#                plt.show()

            # Scale features and make a prediction
#            X = np.hstack((spatial_features, hist_features,
#                           hog_features)).reshape(1, -1)


            
            # DEBUG : Let's try to get feature from single_img_features
            X = single_img_features(img_tosearch[
                ytop:ytop+window, xleft:xleft+window],
                                color_space="YCrCb",
                                hog_channel='ALL'
            ).reshape(1, -1)
            
            test_features = X_scaler.transform(X)
            print(test_features)
            print(np.average(test_features[0]))
            print(np.std(test_features[0]))
            print(test_features.shape)
#            plt.plot([i for i in range(len(test_features[0]))], test_features[0])
#            plt.show()
            # test_features = X_scaler.transform(
            #     np.hstack((shape_feat,
            #                hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            test_prob = svc.predict_proba(test_features)
            print(test_prob)
#            print(test_prediction)
            # Draw Rectangle
            if test_prediction == 1:  # if result is car
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,
                              (xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw,
                               ytop_draw+win_draw+ystart),
                              (0, 0, 255), 6)

    return draw_img


if __name__ == '__main__':
    # training parameter
    dist_pickle = pickle.load(open("svc_pickle.pkl", "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    # paramter
#    ystart = 400
    ystart = 370
    ystop = 656
    scale = 1.0
    cspace = "YCrCb"
    debug_flg = "True"
    hog_channel = 'ALL'

    # debug_flg
    if debug_flg is 'True':
        img = mpimg.imread('test_images/test4.jpg')
#        print("org value average ", np.average(img))
#        input()
        
        print("svc spac :", svc.best_estimator_,
              svc.best_params_, svc.best_score_)
        print("orient :", orient)
        print("pix_per_cell :", pix_per_cell)
        print("cell_per_block :", cell_per_block)
        print("spatial_size :", spatial_size)
        print("hist_bins : ", hist_bins)

        out_img = find_cars(img,  # image file
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

        plt.imshow(out_img)
        plt.show()
