import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Define a function to compute color histogram features
    """

    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


if __name__ == "__main__":
    img = mpimg.imread('test_images/test1.jpg')
#    img = cv2.imread('test_images/test1.jpg')

    rh, gh, bh, bincen, feature_vec = color_hist(img,
                                                 nbins=32,
                                                 bins_range=(0, 256))
    # check the size
    print(np.array(rh).shape)  # [[hist],[indexes]]
    print(np.array(gh).shape)  # [[hist],[indexes]]
    print(np.array(bh).shape)  # [[hist],[indexes]]

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
        plt.show()
    else:
        print('Your function is returning None for at least one variable...')
