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
    Dummy function to simulate processing an image.
    :param image_path: Path to the image file.
    :return: A string indicating result.
    """
    # Implement code here. For now, it's a placeholder.
    return "Result of operation"


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