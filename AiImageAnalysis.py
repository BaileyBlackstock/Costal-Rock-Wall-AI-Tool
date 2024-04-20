import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

colour_ranges = {
    'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
    'yellow': (np.array([20, 70, 50]), np.array([30, 255, 255])),
    'blue': (np.array([100, 70, 50]), np.array([130, 255, 255])),
    'green': (np.array([50, 70, 50]), np.array([70, 255, 255])),
    'white': (np.array([0, 0, 200]), np.array([255, 30, 255]))
}

def plot_grading_curve(rock_masses: list, show: bool= False):
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

def get_real_length(line1: tuple, line2: tuple, line2_real_length: float) -> float:
    line1_mag = np.linalg.norm(np.array(line1[0]) - np.array(line1[1]))
    line2_mag = np.linalg.norm(np.array(line2[0]) - np.array(line2[1]))
    return (line1_mag / line2_mag) * line2_real_length

def get_segmented_images(image: cv2.typing.MatLike, colour_ranges: dict):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    segmented_images = {}
    for colour, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)
        segmented_images[colour] = segmented_image
    
    return segmented_images

def find_contours(image: cv2.typing.MatLike):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(grey, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    """ 
    Filters contours, removing those with a length too short to be a rock border.
    May need to adjust arclength value based on testing (the "> 50" is what im referencing)
    """
    return [contour for contour in contours if cv2.arcLength(contour, True) > 50]

def find_longest_lines(contours) -> list[tuple]:
    longest_lines = []
    for contour in contours:
        hull = cv2.convexHull(contour, returnPoints=False)

        longest_line_length = 0
        longest_line = None

        for i in range(len(hull)):
            for j in range(i+1, len(hull)):
                p1 = tuple(contour[hull[i][0]][0]) 
                p2 = tuple(contour[hull[j][0]][0])

                length = np.linalg.norm(np.array(p1) - np.array(p2))
                if length > longest_line_length:
                    longest_line_length = length
                    longest_line = (p1, p2)

        longest_lines.append(longest_line)
    
    return longest_lines

def get_line_intersect(line1: tuple, line2: tuple):
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

def resize_image(image: cv2.typing.MatLike, target_width: int):
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(image, (target_width, target_height))

def calculate_masses(dimensions: list, density: float, reduction_factor: float) -> list:
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

def measureObject(depth, pixelSizes, imgSize, fov):
    """
    This function converts the width and length in pixels of the objects into real world measurements,
    the unit of measurement is given by the unit of the depth (i.e. if the depth is in mm so is the sizes) 

    :param depth: an integer of the depth of the image
    :param pixelSizes: a list of the width and length of the rocks in pixels
    :param imgSize: the pixel size of the image
    :param fov: the fov of the taken image
    :return: a list of sizes of the rock in real world measurements
    """
    sizes = []

    for x in range(len(pixelSizes)):
        size = []
        for v in range(2):
            size.append(2 * depth * math.tan(fov * (pixelSizes[x][v]/imgSize[v]) / 2))
        sizes.append(size)

    return sizes

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
