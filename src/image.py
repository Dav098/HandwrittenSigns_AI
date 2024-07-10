import cv2
import os
import csv

class Image:
    def __init__(self, image_filename, label):
        self.image_filename = image_filename
        self.label = label

    def load_image(self):
        self.image = cv2.imread(self.image_filename, cv2.IMREAD_GRAYSCALE)

    def crop_image(self):
        leftmost = float('inf')
        rightmost = float('-inf')
        topmost = float('inf')
        bottommost = float('-inf')

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.image[i, j] == 255:
                    if j < leftmost:
                        leftmost = j
                    if j > rightmost:
                        rightmost = j
                    if i < topmost:
                        topmost = i
                    if i > bottommost:
                        bottommost = i

        width = rightmost - leftmost + 1
        height = bottommost - topmost + 1

        # Finding the maximum dimension and adjusting the dimensions to square
        max_dim = max(width, height)
        width_diff = max_dim - width
        height_diff = max_dim - height

        # Extending the sides to make a square
        leftmost -= width_diff // 2
        rightmost += width_diff - (width_diff // 2)
        topmost -= height_diff // 2
        bottommost += height_diff - (height_diff // 2)

        # Making sure the coordinates don't go outside the image
        leftmost = max(0, leftmost)
        rightmost = min(self.image.shape[1], rightmost)
        topmost = max(0, topmost)
        bottommost = min(self.image.shape[0], bottommost)

        # Cut and resize the image
        self.crop_image = self.image[int(topmost):int(bottommost), int(leftmost):int(rightmost)]
        self.crop_image = cv2.resize(self.crop_image, (20, 20))  # Resize do 20x20

        # Adding a 4px black frame
        self.border_image = cv2.copyMakeBorder(self.crop_image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def save_image(self, file_format="jpg"):
        base_filename, _ = os.path.splitext(self.image_filename)

        if base_filename == "predict":
            output_filename = f"{base_filename}.{file_format}"
        else:
            output_filename = f"{base_filename}_{self.get_max_image_count(label=self.label) + 1}.{file_format}"

        cv2.imwrite(output_filename, self.border_image)
        filename_only = os.path.basename(output_filename)

        if base_filename != "predict":
            with open("../data/train.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename_only, self.label])

    def create_new_img(self):
        self.load_image()
        self.crop_image()
        self.save_image()

    def get_max_image_count(self, label=None):
        data_folder = "../data"
        image_files = [f for f in os.listdir(data_folder) if f.endswith(".jpg")]
        if not image_files:
            return 0

        max_count = 0
        for f in image_files:
            try:
                parts = f.split("_")
                number = int(parts[1].split(".")[0])
                if label is not None and parts[0] == str(label):
                    max_count = max(max_count, number)
            except (IndexError, ValueError):
                pass
        return max_count
