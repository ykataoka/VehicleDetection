import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import glob
import pickle as pkl
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# custom module
from spatial_bin import convert_color
from get_hog import get_hog_features
from my_cv_tools import extract_features
from sklearn.model_selection import train_test_split  # v0.18
# from sklearn.cross_validation import train_test_split  # v0.17

debug_flg = False  # plot the graphs
orient = 9  # the number of the direction
pix_per_cell = 8  # the number of pixel to form cell
cell_per_block = 2  # the number of cell to form block


def training(cars, notcars, cspace='YCrCb',
             spatial=32, histbin=32, orientation=9,
             pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
             spatial_feat=True, hist_feat=True, hog_feat=True):

    # feature extraction
    print('feature extraction...')
    car_features = extract_features(cars,
                                    color_space=cspace,
                                    spatial_size=(spatial, spatial),
                                    hist_bins=histbin,
                                    orient=orientation,
                                    pix_per_cell=8,
                                    cell_per_block=2,
                                    hog_channel=hog_channel,
                                    spatial_feat=True,
                                    hist_feat=True,
                                    hog_feat=True)

    notcar_features = extract_features(notcars,
                                       color_space=cspace,
                                       spatial_size=(spatial, spatial),
                                       hist_bins=histbin,
                                       orient=orientation,
                                       pix_per_cell=8,
                                       cell_per_block=2,
                                       hog_channel=hog_channel,
                                       spatial_feat=True,
                                       hist_feat=True,
                                       hog_feat=True)

    X = np.vstack((car_features,
                   notcar_features)).astype(np.float64)

    # normalization
    print('normalization...')
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    print('feature size : ', scaled_X.shape)

    # labels
    print('labels...')
    y = np.hstack((np.ones(len(car_features)),
                   np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # grid search
    parameters = {'kernel': ('linear', 'rbf'),
                  'C': [0.1, 1, 10, 100]}
    svr = svm.SVC(probability=True)
    clf = GridSearchCV(svr, parameters, n_jobs=-1)

    # Training
    print('training (+ grid search)')
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print('time to train :', round(t2-t, 2), '[sec]')
    print('best estimagor :', clf.best_estimator_)
    print('best params :', clf.best_params_)
    print('best scores :', clf.best_score_)

    # Evaluation
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
sh
    # Prediction test
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    t2 = time.time()
    print(round(t2-t, 5), 'Sec to predict', n_predict, 'labels with SVC')

    # save the result
    print('saving to pickle...')
    out_pickle = {"svc": clf,
                  "scaler": X_scaler,
                  "orient": orient,
                  "pix_per_cell": pix_per_cell,
                  "cell_per_block": cell_per_block,
                  "spatial_size": (spatial, spatial),
                  "hist_bins": histbin}

    outfile = open('svc_pickle.pkl', 'wb')
    pkl.dump(out_pickle, outfile)
    print('saving done.')


if __name__ == '__main__':

    """
    read dataset
    """
    # a. mid dataset (total 2321 images)
    cars = glob.glob('../data/test_sample2/vehicle/*')
    notcars = glob.glob('../data/test_sample2/nonvehicle/*')

    # b. small dataset (total 800)
    # images = glob.glob('../data/test_sample2/*')
    # cars, notcars = [], []
    # for image in images:
    #     if 'image' in image or 'extra' in image:
    #         notcars.append(image)
    #     else:
    #         cars.append(image)

    """
    train
    """
    print('training start...')
    training(cars, notcars, cspace='HSV', spatial=32, histbin=32,
             orientation=orient, pix_per_cell=pix_per_cell,
             cell_per_block=cell_per_block, hog_channel='ALL',
             spatial_feat=True, hist_feat=True, hog_feat=True)
    print('training done!')


#    """
#    Test print car and notcar image
#    """
#    if debug_flg is True:
#
#        # read random index
#        car_idx = np.random.randint(0, len(cars))
#        noncar_idx = np.random.randint(0, len(notcars))
#
#        # read images
#        car_img = mpimg.imread(cars[car_idx])  # RGB
#        noncar_img = mpimg.imread(notcars[noncar_idx])  # RGB
#
#        # plot figures
#        fig = plt.figure(figsize=(8, 4))  # needs to be RGB
#        plt.subplot(121)
#        plt.imshow(car_img)
#        plt.title('Example Car Image')
#        plt.subplot(122)
#        plt.imshow(noncar_img)
#        plt.title('Example Non Car Image')
#        fig.tight_layout()
#        plt.savefig('examples/car_not_car.png', dpi=300)
#        # plt.show()

#    """
#    Test print HOG Feature
#    """
#    if debug_flg is True:
#        # read random index
#        car_idx = np.random.randint(0, len(cars))
#        noncar_idx = np.random.randint(0, len(notcars))
#
#        # read images
#        car_img = mpimg.imread(cars[car_idx])  # RGB
#        noncar_img = mpimg.imread(notcars[noncar_idx])  # RGB
#
#        # convert to Gray
#        car_gray = cv2.cvtColor(car_img, cv2.COLOR_RGB2GRAY)
#        noncar_gray = cv2.cvtColor(noncar_img, cv2.COLOR_RGB2GRAY)
#
#        # convert to HSV from RGB
#        hsv_car_img = convert_color(car_img, color_space='HSV',
#                                    size=(64, 64))
#        hsv_noncar_img = convert_color(noncar_img, color_space='HSV',
#                                       size=(64, 64))
#
#        # convert to LUV from RGB
#        luv_car_img = convert_color(car_img, color_space='LUV',
#                                    size=(64, 64))
#        luv_noncar_img = convert_color(noncar_img, color_space='LUV',
#                                       size=(64, 64))
#
#        # convert to HLS from RGB
#        hls_car_img = convert_color(car_img, color_space='HLS',
#                                    size=(64, 64))
#        hls_noncar_img = convert_color(noncar_img, color_space='HLS',
#                                       size=(64, 64))
#
#        # convert to YUV from RGB
#        yuv_car_img = convert_color(car_img, color_space='YUV',
#                                    size=(64, 64))
#        yuv_noncar_img = convert_color(noncar_img, color_space='YUV',
#                                       size=(64, 64))
#
#        # convert to YCrCb from RGB
#        ycrcb_car_img = convert_color(car_img, color_space='YCrCb',
#                                      size=(64, 64))
#        ycrcb_noncar_img = convert_color(noncar_img, color_space='YCrCb',
#                                         size=(64, 64))
#
#        # Define HOG parameters
#
#        # Call our function with vis=True to see an image output
#        features_gray, hog_img_gray = get_hog_features(car_gray, orient,
#                                                       pix_per_cell,
#                                                       cell_per_block,
#                                                       vis=True,
#                                                       feature_vec=False)
#        
#        # HSV - H
#        f_hsv_h, hog_hsv_h = get_hog_features(hsv_car_img[:, :, 0],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#        
#        # HSV - S
#        f_hsv_l, hog_hsv_s = get_hog_features(hsv_car_img[:, :, 1],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # HSV - V
#        f_hsv_s, hog_hsv_v = get_hog_features(hsv_car_img[:, :, 2],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # LUV - L
#        f_luv_h, hog_luv_l = get_hog_features(luv_car_img[:, :, 0],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # LUV - U
#        f_luv_l, hog_luv_u = get_hog_features(luv_car_img[:, :, 1],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # LUV - V
#        f_luv_s, hog_luv_v = get_hog_features(luv_car_img[:, :, 2],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # HLS - H
#        f_hls_h, hog_hls_h = get_hog_features(hls_car_img[:, :, 0],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # HLS - L
#        f_hls_l, hog_hls_l = get_hog_features(hls_car_img[:, :, 1],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # HLS - S
#        f_hls_s, hog_hls_s = get_hog_features(hls_car_img[:, :, 2],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # YUV - Y
#        f_yuv_y, hog_yuv_y = get_hog_features(yuv_car_img[:, :, 0],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # YUV - U
#        f_yuv_u, hog_yuv_u = get_hog_features(yuv_car_img[:, :, 1],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # YUV - V
#        f_yuv_v, hog_yuv_v = get_hog_features(yuv_car_img[:, :, 2],
#                                              orient, pix_per_cell,
#                                              cell_per_block, vis=True,
#                                              feature_vec=False)
#
#        # YCrCb - Y
#        f_ycrcb_y, hog_ycrcb_y = get_hog_features(ycrcb_car_img[:, :, 0],
#                                                  orient, pix_per_cell,
#                                                  cell_per_block, vis=True,
#                                                  feature_vec=False)
#
#        # YCrCb - Cr
#        f_ycrcb_cr, hog_ycrcb_cr = get_hog_features(ycrcb_car_img[:, :, 1],
#                                                    orient, pix_per_cell,
#                                                    cell_per_block, vis=True,
#                                                    feature_vec=False)
#
#        # YCrCb - Cb
#        f_ycrcb_cb, hog_ycrcb_cb = get_hog_features(ycrcb_car_img[:, :, 2],
#                                                    orient, pix_per_cell,
#                                                    cell_per_block, vis=True,
#                                                    feature_vec=False)
#
#        # plot figures
#        f, ((ax1,  ax2,  ax3,  ax4),
#            (ax5,  ax6,  ax7,  ax8),
#            (ax9,  ax10, ax11, ax12),
#            (ax13, ax14, ax15, ax16),
#            (ax17, ax18, ax19, ax20),
#            (ax21, ax22, ax23, ax24),
#            (ax25, ax26, ax27, ax28),
#            (ax29, ax30, ax31, ax32),
#            (ax33, ax34, ax35, ax36),
#            (ax37, ax38, ax39, ax40)) = plt.subplots(10, 4,
#                                                     figsize=(8, 20))
#        f.tight_layout()
#
#        # HSV - ORG
#        ax1.imshow(car_img)
#        ax1.set_title('Original Image', fontsize=10)
#        ax2.imshow(hsv_car_img[:, :, 0], cmap='gray')
#        ax2.set_title('org HSV-H', fontsize=10)
#        ax3.imshow(hsv_car_img[:, :, 1], cmap='gray')
#        ax3.set_title('org HSV-S', fontsize=10)
#        ax4.imshow(hsv_car_img[:, :, 2], cmap='gray')
#        ax4.set_title('org HSV-V', fontsize=10)
#
#        # HSV - HOG
#        ax5.imshow(car_gray, cmap='gray')
#        ax5.set_title('Original Gray Image', fontsize=10)
#        ax6.imshow(hog_hsv_h, cmap='gray')
#        ax6.set_title('HOG HSV-H', fontsize=10)
#        ax7.imshow(hog_hsv_s, cmap='gray')
#        ax7.set_title('HOG HSV-S', fontsize=10)
#        ax8.imshow(hog_hsv_v, cmap='gray')
#        ax8.set_title('HOG HSV-V', fontsize=10)
#
#        # LUV - ORG
#        ax9.imshow(car_img)
#        ax9.set_title('Original Image', fontsize=10)
#        ax10.imshow(luv_car_img[:, :, 0], cmap='gray')
#        ax10.set_title('org LUV-L', fontsize=10)
#        ax11.imshow(luv_car_img[:, :, 1], cmap='gray')
#        ax11.set_title('org LUV-U', fontsize=10)
#        ax12.imshow(luv_car_img[:, :, 2], cmap='gray')
#        ax12.set_title('org LUV-V', fontsize=10)
#
#        # LUV - HOG
#        ax13.imshow(car_gray, cmap='gray')
#        ax13.set_title('Original Gray Image', fontsize=10)
#        ax14.imshow(hog_luv_l, cmap='gray')
#        ax14.set_title('HOG LUV-L', fontsize=10)
#        ax15.imshow(hog_luv_u, cmap='gray')
#        ax15.set_title('HOG LUV-U', fontsize=10)
#        ax16.imshow(hog_luv_v, cmap='gray')
#        ax16.set_title('HOG LUV-V', fontsize=10)
#
#        # HLS - ORG
#        ax17.imshow(car_img)
#        ax17.set_title('Original Image', fontsize=10)
#        ax18.imshow(hls_car_img[:, :, 0], cmap='gray')
#        ax18.set_title('org HLS-H', fontsize=10)
#        ax19.imshow(hls_car_img[:, :, 1], cmap='gray')
#        ax19.set_title('org HLS-L', fontsize=10)
#        ax20.imshow(hls_car_img[:, :, 2], cmap='gray')
#        ax20.set_title('org HLS-S', fontsize=10)
#
#        # HLS - HOG
#        ax21.imshow(car_gray, cmap='gray')
#        ax21.set_title('Original Gray Image', fontsize=10)
#        ax22.imshow(hog_hls_h, cmap='gray')
#        ax22.set_title('HOG HLS-H', fontsize=10)
#        ax23.imshow(hog_hls_l, cmap='gray')
#        ax23.set_title('HOG HLS-L', fontsize=10)
#        ax24.imshow(hog_hls_s, cmap='gray')
#        ax24.set_title('HOG HLS-S', fontsize=10)
#
#        # YUV - ORG
#        ax25.imshow(car_img)
#        ax25.set_title('Original Image', fontsize=10)
#        ax26.imshow(yuv_car_img[:, :, 0], cmap='gray')
#        ax26.set_title('org YUV-Y', fontsize=10)
#        ax27.imshow(yuv_car_img[:, :, 1], cmap='gray')
#        ax27.set_title('org YUV-U', fontsize=10)
#        ax28.imshow(yuv_car_img[:, :, 2], cmap='gray')
#        ax28.set_title('org YUV-V', fontsize=10)
#
#        # YUV - HOG
#        ax29.imshow(car_gray, cmap='gray')
#        ax29.set_title('Original Gray Image', fontsize=10)
#        ax30.imshow(hog_yuv_y, cmap='gray')
#        ax30.set_title('HOG YUV-Y', fontsize=10)
#        ax31.imshow(hog_yuv_u, cmap='gray')
#        ax31.set_title('HOG YUV-U', fontsize=10)
#        ax32.imshow(hog_yuv_v, cmap='gray')
#        ax32.set_title('HOG YUV-V', fontsize=10)
#
#        # YCrCb - ORG
#        ax33.imshow(car_img)
#        ax33.set_title('Original Image', fontsize=10)
#        ax34.imshow(ycrcb_car_img[:, :, 0], cmap='gray')
#        ax34.set_title('org YCrCb-Y', fontsize=10)
#        ax35.imshow(ycrcb_car_img[:, :, 1], cmap='gray')
#        ax35.set_title('org YCrCb-Cr', fontsize=10)
#        ax36.imshow(ycrcb_car_img[:, :, 2], cmap='gray')
#        ax36.set_title('org YCrCb-Cb', fontsize=10)
#
#        # YCrCb - HOG
#        ax37.imshow(car_gray, cmap='gray')
#        ax37.set_title('Original Gray Image', fontsize=10)
#        ax38.imshow(hog_ycrcb_y, cmap='gray')
#        ax38.set_title('HOG YCrCb-Y', fontsize=10)
#        ax39.imshow(hog_ycrcb_cr, cmap='gray')
#        ax39.set_title('HOG YCrCb-Cr', fontsize=10)
#        ax40.imshow(hog_ycrcb_cb, cmap='gray')
#        ax40.set_title('HOG YCrCb-Cb', fontsize=10)
#
#        # overall
#        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
#        plt.savefig('examples/HOG_example2.png', dpi=300)
#        plt.show()
#        plt.close()

#    #
#    if debug_flg is True:
#        # Plot an example of raw and scaled features
#        fig = plt.figure(figsize=(12, 4))
#        plt.subplot(121)
#        plt.imshow(mpimg.imread(cars[car_ind]))
#        plt.title('Original Image')
#        plt.subplot(122)
#        plt.plot(X[car_ind])
#        plt.title('Raw Features')
#        fig.tight_layout()
#        plt.show()
#
#    car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
#                                    hist_bins=32, hist_range=(0, 256))
#
#    notcar_features = extract_features(notcars, cspace='RGB',
#                                       spatial_size=(32, 32),
#                                       hist_bins=32, hist_range=(0, 256))
#
#    if len(car_features) > 0:
#
#        # Create an array stack of feature vectors
#        X = np.vstack((car_features, notcar_features)).astype(np.float64)
#
#        # Fit a per-column scaler
#        X_scaler = StandardScaler().fit(X)
#
#        # Apply the scaler to X
#        scaled_X = X_scaler.transform(X)
#        car_ind = np.random.randint(0, len(cars))
#
#        # Plot an example of raw and scaled features
#        fig = plt.figure(figsize=(12, 4))
#        plt.subplot(131)
#        plt.imshow(mpimg.imread(cars[car_ind]))
#        plt.title('Original Image')
#        plt.subplot(132)
#        plt.plot(X[car_ind])
#        plt.title('Raw Features')
#        plt.subplot(133)
#        plt.plot(scaled_X[car_ind])
#        plt.title('Normalized Features')
#        fig.tight_layout()
#        plt.show()
#    else:
#        print('Your function only returns empty feature vectors...')
