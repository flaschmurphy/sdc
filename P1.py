import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import sys
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    left_lines = {'xx': [], 'yy': []}
    right_lines = {'xx': [], 'yy': []}

    # When deciding on the final lane lines, the y value will always be at the bottom of 
    # the image, meaning the size of the image along the y axis, and the top will always
    # be the same as well - some point just below the horizon
    max_y = img.shape[0]
    min_y = int(img.shape[0] * (3/5))    # just below the horizon

    for line in lines:
        for x1,y1,x2,y2 in line:
	    # The slope of each line indicates if it's part of the left lane or the right. 
	    # One thing to watch out for - since the y coordinate grows down instead of up
	    # a negative slope actually points up and a positive slope points down.
	    # So negative slope lines are left lane, positive are right lane
            slope = ((y2-y1)/(x2-x1))

            # Avoid lines with slope too close to horizontal
            if math.fabs(slope) < 0.5: 
                continue

            if slope > 0: 
                right_lines['xx'].extend([x1, x2])
                right_lines['yy'].extend([y1, y2])
            else:    
                left_lines['xx'].extend([x1, x2])
                left_lines['yy'].extend([y1, y2])

    # To extend the lane lines into a single line for left and a single for right, 
    # we already know the y values (bottom of screen and top of horizon). The problem 
    # is finding x. We can do it using numpy polynomial functions.
    poly_left = np.poly1d(np.polyfit(
       left_lines['yy'],
       left_lines['xx'],
       deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
       right_lines['yy'],
       right_lines['xx'],
       deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    left_bottom = (left_x_start, max_y)
    left_top = (left_x_end, min_y)
    cv2.line(img, left_bottom, left_top, color, thickness)

    right_bottom = (right_x_start, max_y)
    right_top = (right_x_end, min_y)
    cv2.line(img, right_bottom, right_top, color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    if lines is None:
        return

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.


    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def pipeline(image, return_all=False):
    """Processing pipeline for a single image object"""

    pipeline.counter += 1

    # Convert to Grayscale
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    gray_blur = gaussian_blur(gray, kernel_size)

    # Apply Canny
    low_threshold = 50
    high_threshold = 150
    edges = canny(gray_blur, low_threshold, high_threshold)

    # Define the vertices of a region where we expect to find the lanes
    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array([[(0, height), (width*0.5, height*0.59), (width, height)]], dtype=np.int32)

    # Apply the region filter on the canny edges
    cropped = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    rho = 2                  # distance resolution in pixels of the Hough grid
    theta = np.pi/180        # angular resolution in radians of the Hough grid
    threshold = 10           # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20     # minimum number of pixels making up a line
    max_line_gap = 20        # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    hough = hough_lines(cropped, rho, theta, threshold, min_line_length, max_line_gap)
    if hough is None:
        print('*** ERROR: hough transfrom did not return anything')

    #@debug
    #if pipeline.counter % 25 == 0:
    #    plot_all(image, gray, edges, cropped, None, None, pipeline.counter)
    #/debug

    # Draw the lines on the edge image
    line_edges = weighted_img(hough, image)

    if return_all:
        return image, gray, edges, cropped, hough, line_edges
    return line_edges


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = pipeline(image)
    return result


def process_video(path):
    pipeline.counter = 0
    output = 'test_videos_output/' + path.split('/')[-1]
    clip = VideoFileClip(path)
    new_clip = clip.fl_image(process_image)            #NOTE: this function expects color images!!
    new_clip.write_videofile(output, audio=False)
    print('Processed %d frames.' % pipeline.counter)


def plot_all(image, gray, edges, cropped, hough, line_edges, idx):
    """Plot all 6 stages of the pipeline in a single figure"""

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15,15))
    
    ax1.set_title('Original #%d' % idx)
    s = fig.add_subplot(ax1)
    s.imshow(image)

    ax2.set_title('Grayscale #%d' % idx)
    s = fig.add_subplot(ax2)
    s.imshow(gray, cmap='gray')

    ax3.set_title('Canny Edge #%d' % idx)
    s = fig.add_subplot(ax3)
    s.imshow(edges)

    ax4.set_title('Cropped #%d' % idx)
    s = fig.add_subplot(ax4)
    s.imshow(cropped)

    if hough is not None:
        ax5.set_title('Hough #%d' % idx)
        s = fig.add_subplot(ax5)
        s.imshow(hough)

    if line_edges is not None:
        ax6.set_title('Overlay #%d' % idx)
        s = fig.add_subplot(ax6)
        s.imshow(line_edges)

    plt.show()


def main(image_root):
    # Get the long, relative path to each file
    image_paths = [os.path.join(image_root, p) for p in os.listdir(image_root)]

    i = 1
    for image_path in image_paths:
        print('Processing %s' % image_path)
        if image_path.endswith('.mp4'):
            process_video(image_path)
        else:
            image = mpimg.imread(image_path)
            pipeline.counter = 0
            o, g, e, c, h, l = pipeline(image, return_all=True)
            plot_all(o, g, e, c, h, l, i)
            i += 1


if __name__ == '__main__':
    image_root = './test_images'
    if len(sys.argv) > 1:
        image_root = sys.argv[1]
    main(image_root)


