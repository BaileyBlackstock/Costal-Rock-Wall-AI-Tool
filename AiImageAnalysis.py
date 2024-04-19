import os
import cv2
import csv


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


def identify_rock(image_path):
    """
    Dummy function to simulate processing an image.
    :param image_path: Path to the image file.
    :return: A string indicating result.
    """
    # Implement code here. For now, it's a placeholder.
    return "Result of rock identification"

def identify_reference_item(image_path):
    """
    image_path: Path to the image file.
    returns: An array containing an x of y pos of a pixel in the hard hat.

    This function searches the given image to locate the position of the hard hat in the image.
    It works by searching the centre of the image for the pixel closest to orange and returning
    that pixels location.
    
    Prerequisites: the hard hat is in the centre of the image, the hard hat is orange, the hard
    hat is the most orange thing in the centre of the image
    """

    orange = [37, 130, 255]  # colour to compare each pixel to (should be colour of hard hat)

    # shrink the image to 480 pixels long to reduce computation time
    sizeCap = 480
    scale = sizeCap / len(RocksLarge[0])
    Rocks = cv.resize(RocksLarge, (0, 0), fx=scale, fy=scale)
    
    best = 255  # diff between the pixel colour closest to orange
    loc = []  # location of the pixel closest to orange
    
    # search only the middle of the image
    thirdHeight = round(len(Rocks) / 3)
    thirdLength = round(len(Rocks[0]) / 3)
    
    for x in range(thirdHeight, thirdHeight*2):
        for y in range(thirdLength, thirdLength * 2):
            val = Rocks[x][y]
            diff = (abs(val[0] - orange[0]) + abs(val[1] - orange[1]) + abs(val[2] - orange[2])) / 3
            if diff < best:
                best = diff
                loc = [round(x / scale), round(y / scale)]
    
    return loc  # An array containing a x and y positions of a pixel in the hard hat.

def measureObject(depth, pixelSizes, imgSize, fov):
    """
    this function converts the width and length in pixels of the objects into real world measurements

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
