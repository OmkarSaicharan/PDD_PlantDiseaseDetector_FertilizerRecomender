import cv2
import numpy as np
import sys
import os

def find_leaf_contours(image_path, output_path='contour_output.jpg'):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to load image. Check image path: " + image_path)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian blur (removes noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Threshold for binary mask
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 5: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Select one contour (e.g., the largest by area)
    if len(contours) == 0:
        print("No contours found.")
        return output_path, contours
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw ONLY the selected contour in green
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, [largest_contour], -1, (0, 255, 0), 2)

    # Step 7: Save the resulting image
    cv2.imwrite(output_path, contoured_image)

    print(f"Single contour drawn and saved to {output_path}")
    return output_path, [largest_contour]  # Return only the selected contour

if __name__ == "__main__":
    # Usage: python contour.py input_image.jpg optional_output_image.jpg
    if len(sys.argv) < 2:
        print("Usage: python contour.py <input_image> [output_image]")
        sys.exit(1)
    input_image = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else 'contour_output.jpg'
    find_leaf_contours(input_image, output_image)
