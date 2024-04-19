import os
import cv2
import csv

colour_ranges = {
    'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
    'yellow': (np.array([20, 70, 50]), np.array([30, 255, 255])),
    'blue': (np.array([100, 70, 50]), np.array([130, 255, 255])),
    'green': (np.array([50, 70, 50]), np.array([70, 255, 255])),
    'white': (np.array([0, 0, 200]), np.array([255, 30, 255]))
}

#this is a dummy function, im not sure it works, I haven't tested it
def get_real_length(rock_line: tuple, hat_line: tuple, hat_diameter: float) -> float:
    rock_mag = np.linalg.norm(np.array(rock_line[0]) - np.array(rock_line[1]))
    hat_mag = np.linalg.norm(np.array(hat_line[0]) - np.array(hat_line[1]))
    return rock_mag * (hat_mag / hat_diameter)

def segment_images(image: cv2.typing.MatLike, ranges: dict):
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
    return contours

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
