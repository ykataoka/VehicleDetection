import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob
from spatial_bin import bin_spatial  # Color Feature
from color import color_hist  # Color Feature
import pickle as pkl
from get_hog import get_hog_features


def extract_feature_from_img(img, cspace='RGB', spatial_size=(32, 32),
                              hist_bins=32, hist_range=(0, 256),
                              orient=12, pix_per_cell=8, cell_per_block=2):
    """
    @desc : function to extract features from a list of images.
            combine 3 different lists.
    @return size(num_data x (bin_spatial + color_hist))
    """

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else:
        feature_image = np.copy(img)
        
    # 1. RESIZED feature
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image,
                                   color_space=cspace,
                                   size=spatial_size)

    # 2. HISTGRAM feature
    # Apply color_hist() also with a color space option now
    _, _, _, _, hist_features = color_hist(feature_image,
                                           nbins=hist_bins,
                                           bins_range=hist_range)

    # 3. HOG feature
    # Apply get_hog_features() # YUV:0(Y)
    f_yuv_y = get_hog_features(feature_image[:, :, 0],
                               orient, pix_per_cell,
                               cell_per_block, vis=False,
                               feature_vec=False)
    f_yuv_y = f_yuv_y.ravel()  # flatten

    # Return connected feature
    return np.concatenate((spatial_features, hist_features, f_yuv_y))


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256),
                     orient=12, pix_per_cell=8, cell_per_block=2):
    """
    @desc : function to extract features from a list of images.
            combine 3 different lists.
    @return size(num_data x (bin_spatial + color_hist))
    """
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:

        # Read in each one by one
        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(image)

        # Apply bin_spatial() to get spatial color features
        # -> RESIZED feature
        spatial_features = bin_spatial(feature_image,
                                       color_space=cspace,
                                       size=spatial_size)

        # Apply color_hist() also with a color space option now
        # -> HISTGRAM feature
        _, _, _, _, hist_features = color_hist(feature_image,
                                               nbins=hist_bins,
                                               bins_range=hist_range)

        # Apply get_hog_features() # YUV:0(Y)
        # -> HOG feature
        f_yuv_y = get_hog_features(feature_image[:, :, 0],
                                   orient, pix_per_cell,
                                   cell_per_block, vis=False,
                                   feature_vec=False)
        f_yuv_y = f_yuv_y.ravel()  # flatten
        
        # Append the new feature vector to the features list
        # 1.orginal resized + 2.original hisogram + 3.HOG of Y in YUV
        features.append(np.concatenate((spatial_features,
                                        hist_features,
                                        f_yuv_y)))

    # Return list of feature vectors
    return features


if __name__ == '__main__':
    # parameter
    orient = 12  # the number of the direction
    pix_per_cell = 8  # the number of pixel to form cell
    cell_per_block = 2  # the number of cell to form block

    # read images
    images = glob.glob('../data/test_sample/*.jpeg')
    cars = []
    notcars = []
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    car_features = extract_features(cars, cspace='YUV', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256),
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block)

    notcar_features = extract_features(notcars, cspace='YUV',
                                       spatial_size=(32, 32),
                                       hist_bins=32, hist_range=(0, 256),
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block)

    if len(car_features) > 0:

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        print(X.shape)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        output = open('X_scaler.pkl', 'wb')

        # Pickle dictionary using protocol 0.
        pkl.dump(X_scaler, output)

        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))

        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
#        plt.show()
    else:
        print('Your function only returns empty feature vectors...')
