import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import pandas as pd

REAL_HAT_DIAMETER=295
colour_rgb = {
    'red': [[212,0,0],[254,0,0]],
    'yellow': [255, 255, 0],
    'blue': [0, 255, 255], 
    'green': [0, 255, 1],
    'purple': [255,0,254],
}

def get_error(estimated_value, real_value):
    """
    calculates the error between two values.
    :param estimated_value: the predicted value
    :param real_value: the value the prediction is supposed to be
    :return: error (percentage)
    """
    return (abs(real_value - estimated_value) / real_value) * 100
    

def plot_grading_curve(rock_masses: list, show: bool = False):
    """
    Produces a graph cumulative percentage graph of the rock sizes.
    :param rock_masses: the masses of the rocks in kgs
    :param show: whether to show the graph or not
    :return: N/A
    """
    rock_masses_sorted = np.sort(rock_masses)
    total_mass = np.sum(rock_masses_sorted)
    cumulative_masses = np.cumsum(rock_masses_sorted)
    cumulative_percentages = (cumulative_masses / total_mass) * 100

    plt.figure()
    plt.plot(rock_masses_sorted, cumulative_percentages, marker='o')
    plt.xlabel('Rock Size (kg)')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('Rock Grading Curve')
    plt.grid(True)
    if show:
        plt.show()


def imageDepth(referencePixelSize, imgSize, fov):
    """
    This function finds the depth of the reference object by preforming trigonometry
    using the pixel sizes and known hardhat diameter of 295mm

    :param referencePixelSize: reference object diameter
    :param imgSize: image size
    :param fov: fov of camera (~80 degrees for iphone 1x zoom)
    :return: the depth in mm
    """
    HardHatMM = 295/2

    return HardHatMM/math.tan(fov * (referencePixelSize/imgSize[1]) / 2)


def get_segmented_images(image, segmented_image, centers):
    """
    returns a list of images containing only a single colour of the source image
    :param image: original image
    :param segmented_image: the segmented image
    :param centers: a list of colours in the image
    :return: a list of images each containing one of the colours
    """
    # Display the segmented images for each color cluster
    images = []
    for i, color in enumerate(centers):
        mask = cv2.inRange(segmented_image, color, color)
        result = cv2.bitwise_and(image, image, mask=mask)
        images.append((color, result))
    return images


def segment_colors(image, k):
    """
    uses kmeans segmentation to confine the pixels to only a few colours
    :param image: segmented image
    :param k: number of different rock colours
    :return: a list of images each containing only a single image
    """
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Convert to float32
    pixels = np.float32(pixels)

    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and make original image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, centers


def find_contours(image: cv2.typing.MatLike):
    """
    returns large (size > 10) contours in the image
    :param image: an image containing only one colour
    :return: a list of contours in the image
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(grey, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 10
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    return filtered_contours


def find_longest_lines(contours) -> list[tuple]:
    """
    finds the longest line within each contour
    :param contours: a list of contours
    :return: a list of the longest lines
    """
    longest_lines = []
    for contour in contours:
        hull = cv2.convexHull(contour, returnPoints=False)

        longest_line_length = 0
        longest_line = None

        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                p1 = tuple(contour[hull[i][0]][0])
                p2 = tuple(contour[hull[j][0]][0])

                length = np.linalg.norm(np.array(p1) - np.array(p2))
                if length > longest_line_length:
                    longest_line_length = length
                    longest_line = (p1, p2)

        longest_lines.append(longest_line)
        if __debug__:
            print(f"found longest line for a hull at: {longest_lines[-1]}")

    return longest_lines


def get_line_intersect(line1: tuple, line2: tuple):
    """
    finds if two lines intersects
    :param line1: the first line
    :param line2: the second line
    :return: a tuple that is the x and y position of the intersect or none
    """
    dx = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    dy = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(dx, dy)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, dx) / div
    y = det(d, dy) / div

    x_limits1 = (min(line1[0][0], line1[1][0]), max(line1[0][0], line1[1][0]))
    y_limits1 = (min(line1[0][1], line1[1][1]), max(line1[0][1], line1[1][1]))

    x_limits2 = (min(line2[0][0], line2[1][0]), max(line2[0][0], line2[1][0]))
    y_limits2 = (min(line2[0][1], line2[1][1]), max(line2[0][1], line2[1][1]))

    if (x_limits1[0] <= x <= x_limits1[1] and y_limits1[0] <= y <= y_limits1[1]
            and x_limits2[0] <= x <= x_limits2[1] and y_limits2[0] <= y <= y_limits2[1]):
        return (x, y)
    else:
        return None


def find_perp_lines(contours, longest_lines) -> list[tuple]:
    """
    finds the perpendicular lines of each rock in this list
    :param contours: a list of contours
    :param longest_lines: a list of the longest lines in the contours
    :return: a list of perpendicular lines in the rocks
    """
    perp_lines = []

    for contour, line in zip(contours, longest_lines):
        hull = cv2.convexHull(contour)

        mid_x = (line[0][0] + line[1][0]) // 2
        mid_y = (line[0][1] + line[1][1]) // 2

        dx = line[0][0] - mid_x
        dy = line[0][1] - mid_y
        temp = dx
        dx = -dy
        dx = dx * 10 + dx
        dy = temp
        dy = dy * 10 + dy

        perp_line = ((mid_x - dx, mid_y - dy), (mid_x + dx, mid_y + dy))
        temp_line = []

        if __debug__:
            print(f"unadjusted perpendicular line at: {perp_line}")

        for i in range(len(hull)):
            hull_line = (hull[i][0], hull[(i + 1) % len(hull)][0])
            intersection = get_line_intersect(perp_line, hull_line)

            if intersection != None:
                temp_line.append((int(intersection[0]), int(intersection[1])))
                if len(temp_line) == 2:
                    break
        
        if len(temp_line) == 2:
            perp_lines.append(((temp_line[0]), (temp_line[1])))
            if __debug__:
                print(f"found intersections for line, new line is: {perp_lines[-1]}")
        elif __debug__:
            print("failed to find intersections for unadjusted line")

    return perp_lines


def get_rocks_pixel_sizes(image_path, color_count):
    """
    Analyzes an image to segment rocks based on color and measure their sizes. 
    Labels each rock with a unique number from left to right, top to bottom.

    Args:
        image_path (str): The file path to the image to process.
        color_count (int): The number of color clusters to use for k-means segmentation.

    Returns:
        tuple: A tuple containing two elements:
               - A list of rock sizes and id, where each size is a list containing the width and height in pixels and id.
               - The diameter of a reference object in pixels if found.
               If the image cannot be loaded, returns a string indicating the error.
    """
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found"
    id_mapping = {'RockSilhouetteBlack1.jpg': {'area':'Area1', 'real_id': ['J','G','A','F','E','B','C','D']},
            'RockSilhouetteBlack2.jpg': {'area':'Area1', 'real_id': ['J','G','A','F','B']},
            'RockSilhouetteBlack3.jpg': {'area':'Area2','real_id': ['M','J','I','A','G','F','B']},
            'RockSilhouetteBlack4.jpg': {'area':'Area2','real_id': ['M','L','A','G','B']},
            'RockSilhouetteBlack5.jpg': {'area':'Area2','real_id': ['M','J','I','A','G','B']}
            }

    original_image = image.copy()
    segmented_image, centers = segment_colors(image, color_count + 2)
    segmented = get_segmented_images(image, segmented_image, centers)

    rocks_with_positions = []
    rock_sizes_id=[]
    reference_diameter = 0
    base_filename = os.path.basename(image_path)
    folder_name = 'label'
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Process each segmented image to find contours and measure rocks.
    for color, segment_image in segmented:
        if np.all(color < 20):  # Skip dark colors.
            continue

        contours = find_contours(segment_image)
        longest_lines = find_longest_lines(contours)
        perpendicular_lines = find_perp_lines(contours, longest_lines)

        for line, contour, width, height in zip(longest_lines, contours, longest_lines, perpendicular_lines):
            min_x = min(line[0][0], line[1][0])  # Get the minimum x value from the longest line
            diameter_length = min(dis(width),dis(height))

            if diameter_length<100:
                continue  # Skip smaller rocks.
            if np.all(color > 235):  # Use a specific color to set reference diameter.
                reference_diameter = dis(width)
                # Draw reference object diameter.
                cv2.line(image, width[0], width[1], (255,0,254), 4)
            else:
                rocks_with_positions.append((min_x, width, height, contour, color))
    # Sort rocks by their leftmost x-coordinate from the longest line.
    rocks_with_positions.sort()

    # Re-label rocks based on sorted order
    rock_count = 0
    for min_x, width, height, contour, color in rocks_with_positions:
        # Check if the image has mapping in id_mapping
        image_filename = os.path.basename(image_path)
        if image_filename in id_mapping:
            real_ids = id_mapping[image_filename]['real_id']
            # Check if rock_count is within the range of real_ids
            if rock_count < len(real_ids):
                # Use real_id for labeling
                rock_id = real_ids[rock_count]
            else:
                # Continue using original rock_count if there's no corresponding real_id
                rock_id = str(rock_count)
        else:
            # Continue using original rock_count if the image doesn't have mapping in id_mapping
            rock_id = str(rock_count)

        rock_sizes_id.append([dis(width), dis(height), rock_id])
        intersection_point = get_intersection_point(width, height)
        rock_number_position = (int(intersection_point[0]), int(intersection_point[1]))
        cv2.line(original_image, width[0], width[1], (255,255,255), 2)
        cv2.line(original_image, height[0], height[1], (255,255,255), 2)
        cv2.putText(original_image, rock_id, rock_number_position, cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        rock_count += 1

    cv2.imwrite(f'{folder_name}/numbered_{base_filename}', original_image)
    return rock_sizes_id, reference_diameter


def resize_image(image: cv2.typing.MatLike, target_width: int):
    """
    resize the image
    :param image: original image
    :param target_width: new width to adjust to
    :return: new image of target size
    """
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(image, (target_width, target_height))


def calculate_masses(dimensions: list, density: float, reduction_factor: float) -> list:
    # converts dimensions to masses
    return [((width * height * height) * reduction_factor) * density for width, height in dimensions]

def calculate_volumes(dimensions: list, reduction_factor: float) -> list:
    # converts dimensions to only volumes when density is not available
    return [((width * height * height) * reduction_factor) for width, height in dimensions]

def count_large_color_areas(image, area_threshold):
    """
    Count the number of large areas with specified colors in the image.

    Args:
        image (numpy.ndarray): Input image.
        area_threshold (int): Minimum area threshold for considering an area as large.

    Returns:
        int: Number of large areas detected for each color.
    """
    colour_rgb = {
        'red': [[212, 0, 0], [254, 0, 0]],
        'yellow': [255, 255, 0],
        'blue': [0, 255, 255],
        'green': [0, 255, 1],
        'purple': [255, 0, 254],
    }

    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize a set to store unique colors with large areas
    unique_colors_with_large_areas = set()

    # Iterate over each color
    for color, color_range in colour_rgb.items():
        # Convert color range to numpy array
        if isinstance(color_range[0], list):
            color_ranges = [np.array(lower, dtype=np.uint8) for lower in color_range]
            lowerb = color_ranges[0]
            upperb = color_ranges[1]
        else:
            lowerb = np.array(color_range, dtype=np.uint8)
            upperb = np.array(color_range, dtype=np.uint8)

        # Mask areas with the specified color
        mask = cv2.inRange(image_rgb, lowerb, upperb)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

        # Check if large areas exist for this color
        if len(large_contours) > 0:
            unique_colors_with_large_areas.add(color)

    return len(unique_colors_with_large_areas)

def resize_image(image: cv2.typing.MatLike, target_width: int):
    """
    resize the image
    :param image: original image
    :param target_width: new width to adjust to
    :return: new image of target size
    """
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(image, (target_width, target_height))

def calculate_masses(dimensions: list, density: float, reduction_factor: float) -> list:
    # converts dimensions to masses
    return [((width * height * height) * reduction_factor) * density for width, height in dimensions]

def read_image_directory(directory_path):
    """
    Reads a directory containing image files, displays each image, and returns a list of image paths.
    :param directory_path: Path to the directory containing images.
    :return: List of image paths.
    """
    supported_formats = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(supported_formats)]
    '''
    # Create a resizable window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # Resize the window to a fixed size
    cv2.resizeWindow('Image', 640, 480)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        cv2.imshow('Image', image)
        cv2.waitKey(0)  # Wait for any key to be pressed to proceed

    cv2.destroyAllWindows()  # Close the image window when done
    '''
    return image_paths

def crop_edges(image_path, output_path, crop_amount):
    """
    Crop the edges of an image by the specified amount.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the output cropped image.
        crop_amount (int): Amount to crop from each edge.
    """
    image = Image.open(image_path)
    width, height = image.size

    # Calculate the crop box dimensions
    left = crop_amount
    top = crop_amount
    right = width - crop_amount
    bottom = height - crop_amount

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Save the cropped image
    cropped_image.save(output_path)

def batch_crop_images(input_folder, output_folder, crop_amount):
    """
    Batch crop all images in a folder by the specified amount.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to save the output cropped images.
        crop_amount (int): Amount to crop from each edge.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each image file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            crop_edges(input_path, output_path, crop_amount)

def output_to_csv(data, output_file_path):
    """
    Outputs data to a CSV file.
    :param data: List of tuples containing data to write to CSV. Each tuple should be (image_path, id, length, width, depth, volume)
    :param output_file_path: Path to the output CSV file.
    """
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Unique_ID", "Length (mm)", "Width (mm)", "Depth (mm)", "Volume (m3)"])
                     
        for entry in data:
            image_path, id, length, width, depth, volume ,_,_,_,_,_,_,_,_= entry
            writer.writerow([os.path.basename(image_path), id, length, width, depth, volume])
        writer.writerow(["File Name","","Mean Length (mm)", "Mean Width (mm)", "Mean Depth (mm)", "Mean Volume (m3)"
                     ])
        image_path,_,_,_,_,_,mean_length, mean_width, mean_width, mean_volume,median_length, median_width, median_width, median_volume=data[0]

        writer.writerow([os.path.basename(image_path),"",mean_length, mean_width, mean_width, mean_volume])
     
        writer.writerow(["File Name","","Median Length (mm)", "Median Width (mm)", "Median Depth (mm)", "Median Volume (m3)"])
        writer.writerow([os.path.basename(image_path),"",median_length, median_width, median_width, median_volume])


def get_rock_measurements(depth, pixelSizes, imgSize, fov):
    """
    This function converts the width and length in pixels of the objects into real world measurements,
    the unit of measurement is given in meters

    :param depth: a float of the depth of the image in mm
    :param pixelSizes: a list of the width and length of the rocks in pixels
    :param imgSize: the pixel size of the image
    :param fov: the fov of the taken image
    :return: a list of sizes of the rock in real world measurements
    """
    sizes = []

    for x in range(len(pixelSizes)):
        size = []
        for v in range(2):
            size.append(2 * depth * math.tan(fov * (pixelSizes[x][v] / imgSize[v]) / 2)/1000)
        sizes.append(size)

    return sizes


def dis(line):
    # finds and returns the length of a line
    x1, y1 = line[0]
    x2, y2 = line[1]
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def get_real_length(rock_size_id: list, hat_pixel_diameter: float, hat_real_diameter: float) -> tuple:
    """
    Finds the real world dimensions of a rock given the pixel dimensions and the reference object's real size.
    :param rock_size: Tuple containing the pixel lengths of the rock's longest and perpendicular diameters.
    :param hat_pixel_diameter: Pixel diameter of the reference object (safety helmet).
    :param hat_real_diameter: Real world diameter of the reference object.
    :return: Tuple containing the real world lengths of the rock's longest and perpendicular diameters.
    """
    real_long = (rock_size_id[0] / hat_pixel_diameter) * hat_real_diameter
    real_perp = (real_long / rock_size_id[0]) * rock_size_id[1]   
    return (real_long, real_perp, rock_size_id[2])

def count_large_color_areas(image, colour_rgb, area_threshold):
    """
    Count the number of large areas with specified colors in the image.

    Args:
        image (numpy.ndarray): Input image.
        colour_rgb (dict): Dictionary containing color names and their corresponding RGB values.
        area_threshold (int): Minimum area threshold for considering an area as large.

    Returns:
        int: Number of large areas detected for each color.
    """
    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize a set to store unique colors with large areas
    unique_colors_with_large_areas = set()

    # Iterate over each color
    for color, color_range in colour_rgb.items():
        # Convert color range to numpy array
        if isinstance(color_range[0], list):
            color_ranges = [np.array(lower, dtype=np.uint8) for lower in color_range]
            lowerb = color_ranges[0]
            upperb = color_ranges[1]
        else:
            lowerb = np.array(color_range, dtype=np.uint8)
            upperb = np.array(color_range, dtype=np.uint8)

        # Mask areas with the specified color
        mask = cv2.inRange(image_rgb, lowerb, upperb)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

        # Check if large areas exist for this color
        if len(large_contours) > 0:
            unique_colors_with_large_areas.add(color)

    return len(unique_colors_with_large_areas)

def detect_large_color_areas(image_path, colour_rgb, area_threshold):
    """
    Detect large areas with specified colors in the image.

    Args:
        image_path (str): Path to the input image file.
        colour_rgb (dict): Dictionary containing color names and their corresponding RGB values.
        area_threshold (int): Minimum area threshold for considering an area as large.

    Returns:
        int: Number of large areas detected for each color.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Count the number of large color areas
    large_color_area_counts = count_large_color_areas(image, colour_rgb, area_threshold)

    return large_color_area_counts


def detect_large_color_areas(image_path, colour_rgb, area_threshold):
    """
    Detect large areas with specified colors in the image.

    Args:
        image_path (str): Path to the input image file.
        colour_rgb (dict): Dictionary containing color names and their corresponding RGB values.
        area_threshold (int): Minimum area threshold for considering an area as large.

    Returns:
        int: Number of large areas detected for each color.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Count the number of large color areas
    large_color_area_counts = count_large_color_areas(image, colour_rgb, area_threshold)

    return large_color_area_counts

def calculate_volume(rock_sizes, reduction_factor=0.05):
    """
    Calculates the approximate volumes of rocks given their measured dimensions and
    applies a reduction factor to account for irregular shapes.

    :param rock_sizes: List of tuples, each containing the real world longest and perpendicular diameters of a rock.
    :param reduction_factor: Percentage to reduce from the calculated volume to account for irregularities.
    :return: List of volumes after applying the reduction factor.
    """
    volumes = []
    for (long_diameter, perp_diameter) in rock_sizes:
        # Assume the depth is the same as the perpendicular diameter (smallest of the two measurements)
        depth = perp_diameter
        
        # Calculate the volume of the rock assuming it's roughly a cuboid
        raw_volume = long_diameter * perp_diameter * depth/1e9
        
        # Apply the reduction factor to account for irregular shape
        adjusted_volume = raw_volume * (1 - reduction_factor)
        
        volumes.append(adjusted_volume)
    
    return volumes

def main():
    # Define the parameters for the project
    image_directory = "AI Image Analysis Data"
    output_file_suffix = "_output.csv"
    fov = 80 * math.pi / 180  # Convert field of view to radians
    target_width = 800  # Resize target width for images
    reduction_factor = 0.9  # Assuming a simplistic model for rock mass estimation
    area_threshold = 500 # Minimum area threshold for considering an area as large

    # Read images from directory
    image_paths = read_image_directory(image_directory)

    # List to store all measurements data
    measurements_data = []
    rock_masses = []  # List to collect rock masses for grading curve

    for image_path in image_paths:
        # Prompt user to input density
        '''
        print('Please enter the density, or there will be only volumes.')
        density = input('density: ')  # Density of rocks, assuming granite in kg/m^3
        '''
        # Suppose no density is input
        density = ''

        size_id_result = identify_rock(image_path)  
        size_result = [[row[0], row[1]] for row in size_id_result]
        volume_result = calculate_volume(size_result, reduction_factor=0.05) 
        
        # Calculate median and mean of length, width, and volume
        lengths = [row[0] for row in size_id_result]
        widths = [row[1] for row in size_id_result]
        volumes = volume_result
        
        median_length = np.median(lengths)
        median_width = np.median(widths)
        median_volume = np.median(volumes)
        
        mean_length = np.mean(lengths)
        mean_width = np.mean(widths)
        mean_volume = np.mean(volumes)

        # Load and resize image
        image = cv2.imread(image_path)
        resized_image = resize_image(image, target_width)

        rock_colors_number = count_large_color_areas(resized_image, area_threshold)

        # Get the contours, sizes of rocks in pixels, and hat diameter
        pixel_sizes, hat_diameter = get_rocks_pixel_sizes(resized_image, rock_colors_number)

        # If no valid hat_diameter is found, continue to next image or use a fallback value
        if hat_diameter == 0:
            print(f"Error: No valid reference object found in {image_path}. This image will be skipped.")
            continue

        # Estimate depth
        depth = imageDepth(hat_diameter, resized_image.shape, fov)

        # Convert pixel sizes to real-world measurements
        real_sizes = get_rock_measurements(depth, pixel_sizes, resized_image.shape, fov)

        if density == '':
            # Calculate rock volumes based on their dimensions if there is no density
            masses = calculate_volumes(real_sizes, reduction_factor)
        else:
            density = int(density)
            # Calculate rock masses based on their dimensions
            masses = calculate_masses(real_sizes, density, reduction_factor)

        rock_masses.extend(masses)  # Collect all rock masses for grading curve

        # Generate data for CSV output
        #for i, dimensions in enumerate(real_sizes):
            #width, Height = dimensions
            #mass = masses[i]
           #measurements_data.append(
                #(image_path, i, width * 1000, Height * 1000, mass))  # converting meters to mm

        # Extract the base name without the extension
        base_name = os.path.splitext(image_path)[0]

        # Append the output file suffix
        output_file_path = base_name + output_file_suffix

        # Output the data to a CSV file
        output_to_csv(measurements_data, output_file_path)

        # Plot the grading curve using the collected rock masses
        plot_grading_curve(rock_masses, show=True)

        print("Data processing complete. Results saved to:", output_file_path)


if __name__ == "__main__":
    main()
