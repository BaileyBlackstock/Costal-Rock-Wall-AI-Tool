import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import math


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
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

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

    return longest_lines


def get_line_intersect(line1: tuple, line2: tuple):
    """
    finds if two lines intersects
    :param line1: the first line
    :param line2: the second line
    :return: a bool that states if the lines do or don't intersect
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
    y_limtit2 = (min(line2[0][1], line2[1][1]), max(line2[0][1], line2[1][1]))

    if (x_limits1[0] <= x <= x_limits1[1] and y_limits1[0] <= y <= y_limits1[1]
            and x_limits2[0] <= x <= x_limits2[1] and y_limtit2[0] <= y <= y_limtit2[1]):
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
        dx = dx * 0.1 + dx
        dy = temp
        dy = dy * 0.1 + dy

        perp_line = ((mid_x - dx, mid_y - dy), (mid_x + dx, mid_y + dy))
        temp_line = []

        for i in range(len(hull)):
            hull_line = (hull[i][0], hull[(i + 1) % len(hull)][0])
            intersection = get_line_intersect(perp_line, hull_line)

            if intersection != None:
                temp_line.append((int(intersection[0]), int(intersection[1])))
                if len(temp_line) == 2:
                    break

        perp_lines.append(((temp_line[0]), (temp_line[1])))

    return perp_lines


def get_rocks_pixel_sizes(image, colourCount):
    """
    finds the pixel lengths of the rocks in the image
    :param image: the silhouette image of the rocks
    :param colourCount: the number of unique rock colours
    :return: a 2d list of rock pixel sizes
    """
    # colour segment the image to remove noise
    segmented_image, centers = segment_colors(image, colourCount + 2)

    # separate into multiple images of each colour
    segmented = get_segmented_images(image, segmented_image, centers)

    rockSizes = []
    referenceDiameter = 0

    for colour, segment_image in segmented:
        if colour[0] < 20 and colour[1] < 20 and colour[2] < 20:
            continue  # skip the colour black

        contours = find_contours(segment_image)
        longs = find_longest_lines(contours)
        perp = find_perp_lines(contours, longs)

        if colour[0] > 235 and colour[1] > 235 and colour[2] > 235:
            referenceDiameter = dis(longs[0])
        else:
            for i in range(len(longs)):
                newSize = [dis(longs[i]), dis(perp[i])]
                rockSizes.append(newSize)

    return rockSizes, referenceDiameter


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

    # Create a resizable window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # Resize the window to a fixed size
    cv2.resizeWindow('Image', 640, 480)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        cv2.imshow('Image', image)
        cv2.waitKey(0)  # Wait for any key to be pressed to proceed

    cv2.destroyAllWindows()  # Close the image window when done

    return image_paths


def output_to_csv(data, output_file_path):
    """
    Outputs data to a CSV file.
    :param data: List of tuples containing data to write to CSV.
    :param output_file_path: Path to the output CSV file.
    """
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "OperationResult"])
        for row in data:
            writer.writerow(row)


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
    

def main():
    directory_path = "AI Image Analysis Data"  # Update this path
    output_file_suffix = "_output.csv"  # Update this path as needed

    image_paths = read_image_directory(directory_path)
    results = []
    reference_results = []

    for image_path in image_paths:
        # Process each image and store the rock analysis result
        result = identify_rock(image_path)
        results.append((image_path, result))
        # Process each image and store the reference result
        reference_result = identify_reference_item(image_path)
        reference_results.append((image_path, reference_result))
        # For each image, output a csv file
        output_file_path = ''
        output_file_path = image_path.replace('.jpg','').replace('.png','') + output_file_suffix
        output_to_csv(results, output_file_path)
        print(f"Analysis complete. Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
