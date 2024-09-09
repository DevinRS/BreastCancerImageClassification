import cv2
import matplotlib.pyplot as plt
import numpy as np

def canny_count(image_file, blur_kernel_size=(11,11), low_threshold=50, high_threshold=150, morph_shape=cv2.MORPH_ELLIPSE, dilation_kernel_size=(5, 5), dilation_iterations=2):
    # Read the image file
    image = cv2.imread(image_file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Conversion of image Into Gray Color
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)    # Otsu's thresholding
    blur = cv2.GaussianBlur(binary, blur_kernel_size, 0)    # Add Gaussian Blur
    canny = cv2.Canny(blur, low_threshold, high_threshold, 3)    # Using Canny Edge Detection Algorithm
    dilated = cv2.dilate(canny, cv2.getStructuringElement(morph_shape, dilation_kernel_size), iterations=dilation_iterations)    # Dilation is used to observe changes
    (cnt, heirarchy ) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_contour_area = 100  # Adjust based on your nuclei size
    filtered_cnt = [c for c in cnt if cv2.contourArea(c) > min_contour_area]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)

    return image, rgb, len(cnt)

def hough_circle_count(image_file, blur_kernel_size=(11,11), dp=1.0, min_dist=50, param1=150, param2=11, min_radius=40, max_radius=60):
    # Read the image file
    image = cv2.imread(image_file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)    # Otsu's thresholding
    blur = cv2.GaussianBlur(binary, blur_kernel_size, 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    # Check if circles were found
    if circles is not None:
        # Convert the circle parameters (x, y, r) to integers
        circles = np.round(circles[0, :]).astype("int")

        # Draw the circles on the original image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for (x, y, r) in circles:
            cv2.circle(rgb_image, (x, y), r, (0, 255, 0), 4)  # Draw the outer circle
            cv2.circle(rgb_image, (x, y), 2, (0, 0, 255), 3)  # Draw the center of the circle

        circle_count = len(circles)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        circle_count = 0

    return image, rgb_image, circle_count

def watershed_detection(image_file, kernel_size=(3, 3), iterations_opening=1, iterations_dilation=2, dist_transform_mask_size=5, dist_transform_threshold=0.2):
    # Read the image file
    image = cv2.imread(image_file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise with morphological operations
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations_opening)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=iterations_dilation)

    # Finding sure foreground area using distance transform and thresholding
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, dist_transform_mask_size)
    _, sure_fg = cv2.threshold(dist_transform, dist_transform_threshold * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark the boundaries with red color

#   if show_img:
#     # Plot original image, binary image, and markers overlaid on the original image in one subplot
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     axs[0].set_title('Original Image')
#     axs[0].axis('off')
#     axs[1].imshow(binary, cmap='gray')
#     axs[1].set_title('Binary Image')
#     axs[1].axis('off')

    # Create a color map with a single color for markers
    marker_color = [0, 255, 0]  # Green color for all markers
    color_map = np.zeros_like(image)
    color_map[markers > 1] = marker_color

    # Convert the color_map from BGR to RGB (for Matplotlib)
    rgb_color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Combine the color map with the original image
    overlay = cv2.addWeighted(image, 0.7, rgb_color_map, 0.3, 0)

    # axs[2].imshow(overlay)
    # axs[2].set_title('Markers Overlaid')
    # axs[2].axis('off')
    # plt.tight_layout()
    # plt.show()

    return image, overlay, np.max(markers) - 1