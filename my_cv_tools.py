import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt  # for debug


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    @param pix_per_cell :
    @param cell_per_block :
    @param vis : True if image is require
    @param feature_vec :
    @return HOG image (Histogram of Orientation Gradient)
    """
    # Call with two outputs if vis==True
    if vis is True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell,
                                                   pix_per_cell),
                                  cells_per_block=(cell_per_block,
                                                   cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell,
                                        pix_per_cell),
                       cells_per_block=(cell_per_block,
                                        cell_per_block),
                       transform_sqrt=True,
                       visualise=vis,
                       feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    """
    @desc compute binned color features
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    @desc  compute color histogram features
    @note  NEED TO CHANGE bins_range if reading .png files with mpimg.
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0],
                                 bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1],
                                 bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2],
                                 bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    @color_space : for all of feature computation
    @hog_channel : "ALL" for 3 channels, or 0, 1, 2 to specify channel
    @spatial_feat : include spatial_feat of color_space or not
    @hist_feat : include histogram of color_space or not
    @hog_feat : include hog of color_space or not
    @desc extract features from a list of images for training
    """
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        file_features = []

        # Read in each one by one
        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # 1. spatial feature include?
        if spatial_feat is True:
            spatial_features = bin_spatial(feature_image,
                                           size=spatial_size)
            file_features.append(spatial_features)

        # 2. histogram feature include?
        if hist_feat is True:
            hist_features = color_hist(feature_image,
                                       nbins=hist_bins)
            file_features.append(hist_features)

        # 3. hog feature include?
        if hog_feat is True:
            if hog_channel == 'ALL':
                tmp = []
                for channel in range(feature_image.shape[2]):
                    tmp.append(get_hog_features(feature_image[:, :, channel],
                                                orient, pix_per_cell,
                                                cell_per_block,
                                                vis=False, feature_vec=True))
                hog_features = np.ravel(tmp)
            else:
                tmp = get_hog_features(feature_image[:, :, hog_channel],
                                       orient, pix_per_cell,
                                       cell_per_block,
                                       vis=False, feature_vec=True)
                hog_features = tmp
            file_features.append(hog_features)

        # Append the feature vector to the features list
        features.append(np.concatenate(file_features))

    # Return list of feature vectors
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    @desc returns the list of the window,
          which start and stop positions in both x and y,
          with window size and overlap fraction
    """

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    @desc Define a function to draw bounding boxes
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    @desc extract features from "a single image window" for prediction.
          This function is very similar to extract_features().
    """
    # Define an empty list to receive features
    img_features = []

    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    # Compute spatial features if flag is set
    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    # Compute histogram features if flag is set
    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    # Compute HOG features if flag is set
    if hog_feat is True:
        if hog_channel is 'ALL':
            hog_features = []
            for c in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, c],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient, pix_per_cell,
                                            cell_per_block,
                                            vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """
    @param windows : list of windows (start&stop of x&y)
    @desc apply car prediction to windows
    @return car location
    """
    # Create an empty list to receive positive detection windows
    on_windows = []

    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]],
                              (64, 64))

        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict using your classifier
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window
        if prediction == 0:
            on_windows.append(window)

    # Return windows for positive detections
    return on_windows


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#            elif color_space == 'YUV':
#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
#            elif color_space == 'YCrCb':
#                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# def convert_color(img, color_space='RGB', size=(32, 32)):
#     """
#     @desc compute color histogram features
#     """
#
#     # Convert image to new color space (if specified)
#     if color_space != 'RGB':
#         if color_space == 'HSV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#         elif color_space == 'LUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
#         elif color_space == 'HLS':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#         elif color_space == 'YUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#         elif color_space == 'YCrCb':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
#     else:
#         feature_image = np.copy(img)
#
#     # Use cv2.resize().ravel() to create the feature vector
#     features = cv2.resize(feature_image, size)
#
#     # Return the feature vector
#     return features
