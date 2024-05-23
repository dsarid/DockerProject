from pathlib import Path
from matplotlib.image import imread, imsave
import random


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class Img:

    def __init__(self, path):
        """
        Do not change the constructor implementation
        """
        self.path = Path(path)
        self.data = rgb2gray(imread(path)).tolist()

    def save_img(self):
        """
        Do not change the below implementation
        """
        new_path = self.path.with_name(self.path.stem + '_filtered' + self.path.suffix)
        imsave(new_path, self.data, cmap='gray')
        return new_path

    def blur(self, blur_level=16):

        height = len(self.data)
        width = len(self.data[0])
        filter_sum = blur_level ** 2

        result = []
        for i in range(height - blur_level + 1):
            row_result = []
            for j in range(width - blur_level + 1):
                sub_matrix = [row[j:j + blur_level] for row in self.data[i:i + blur_level]]
                average = sum(sum(sub_row) for sub_row in sub_matrix) // filter_sum
                row_result.append(average)
            result.append(row_result)

        self.data = result

    def contour(self):
        for i, row in enumerate(self.data):
            res = []
            for j in range(1, len(row)):
                res.append(abs(row[j-1] - row[j]))

            self.data[i] = res

    def rotate(self):
        result = list(zip(*reversed(self.data)))
        self.data = result

    def salt_n_pepper(self):
        rand_val = 0
        for i in range(len(self.data)):
            for g in range(len(self.data[i])):
                rand_val = random.random()
                if rand_val < 0.2:
                    # print(f"{rand_val}: is less than 0.2 ")
                    self.data[i][g] = 255
                elif rand_val > 0.8:
                    self.data[i][g] = 0

    def concat(self, other_img, direction='horizontal'):
        res = []
        if direction == "horizontal":
            if len(self.data) == len(other_img.data):
                for i in range(len(self.data)):
                    res.append(self.data[i] + other_img.data[i])
                self.data = res
            else:
                raise RuntimeError("Dimensions not matched!")
        else:
            raise RuntimeError("Unknown direction")

    def segment(self):
        for i in range(len(self.data)):
            for g in range(len(self.data[i])):
                if self.data[i][g] > 100:
                    self.data[i][g] = 255
                else:
                    self.data[i][g] = 0

