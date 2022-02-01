import numpy as np
import base64
from PIL import Image
from io import BytesIO

from Convert import ConvertHeader as ConvH


class Picture:

    def __init__(self):
        self._type_of_transformation = None

        self._origin_b64 = None
        self._origin_array = None

        self._result_array = None
        self._result_b64 = None

    @property
    def type_of_transformation(self):
        if not self._type_of_transformation:
            raise TypeError("Type o transformation  is undefined!")
        return self._type_of_transformation

    @type_of_transformation.setter
    def type_of_transformation(self, received_type):
        self._type_of_transformation = received_type

    @property
    def origin_b64(self):
        if not self._origin_b64:
            raise TypeError("String of source image not found!")
        return self._origin_b64

    @origin_b64.setter
    def origin_b64(self, received_b64_string):
        self._origin_b64 = received_b64_string

    def get_origin_array(self):
        if self._origin_array is None:
            raise TypeError("Array not found!")
        return self._origin_array

    def set_origin_array(self):

        if not self.origin_b64:
            raise TypeError("String of source image not found!")

        image = Picture.b64_decode(self.origin_b64)

        if image.mode == "RGB":
            channels = 3
        elif image.mode == "L":
            channels = 1
        elif image.mode == "RGBA":
            image = Picture.rgba2rgb(image)
            channels = 3
        # elif image.mode == "P":
        #     image.convert("RGB")
        #
        #     channels = 3
        else:
            print("Unknown mode: %s" % image.mode)
            return

        height, width = image.size
        pixel_values = list(image.getdata())
        pixel_values = np.array(pixel_values).reshape((width, height, channels))
        self._origin_array = pixel_values


    def get_result_array(self):
        return Image.fromarray(self._result_array.astype('uint8'), 'RGB')

    def set_result_array(self):
        result_array = []
        type_of_transformation = self.type_of_transformation.lower()
        source_array = self.get_origin_array()
        if type_of_transformation == "xyz":
            result_array = ConvH.xyz(source_array)
        elif type_of_transformation == "cie_lab":
            result_array = ConvH.cie_lab(source_array)
        elif type_of_transformation == "hunter_lab":
            result_array = ConvH.hunter_lab(source_array)
        elif type_of_transformation == "hsv":
            result_array = ConvH.hsv(source_array)
        elif type_of_transformation == "hsl":
            result_array = ConvH.hsl(source_array)
        elif type_of_transformation == "luv":
            result_array = ConvH.luv(source_array)
        elif type_of_transformation == "cmy":
            result_array = ConvH.cmy(source_array)
        elif type_of_transformation == "yxy":
            result_array = ConvH.yxy(source_array)

        self._result_array = result_array

    def get_result_b64(self):
        if not self._result_b64:
            raise TypeError("String of source image not found!")
        return self._result_b64

    def set_result_b64(self):
        
        self._result_b64 = Picture.b64_encode(self._result_array)

    @staticmethod
    def b64_decode(encoded_string):
        image = Image.open(BytesIO(base64.b64decode(encoded_string)))
        return image

    @staticmethod
    def b64_encode(array_of_image):
        image = Image.fromarray(array_of_image.astype('uint8'), 'RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        b64_string = base64.b64encode(buffered.getvalue())
        return b64_string

    @staticmethod
    def rgba2rgb(rgba_image):
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask=rgba_image.split()[3])
        background.save("sample_2.jpeg", "JPEG", quality=100)
        # rgb_image = Image.open("sample_2.jpeg")

        return background


def main(received_base64_string, type_of_transformation):
    if type_of_transformation.lower() not in ["xyz", "cie_lab", "hunter_lab", "hsv", "hsl", "luv", "cmy", "yxy"]:
        raise ValueError("Unexpected conversion type")
    picture = Picture()
    picture.origin_b64 = received_base64_string
    picture.type_of_transformation = type_of_transformation

    picture.set_origin_array()
    picture.set_result_array()

    picture.set_result_b64()
    result_base64_string = picture.get_result_b64()

    return result_base64_string


if __name__ == "__main__":
    # pass
    import matplotlib.pyplot as plt

    with open("image1.png", "rb") as image_file:
        b64_string = base64.b64encode(image_file.read())

    b64_res = main(b64_string, "hunter_lab")
    image_res = Picture.b64_decode(b64_res)
    fig = plt.figure(figsize=(10, 5))

    ax2 = fig.add_subplot()

    ax2.imshow(image_res)
    plt.show()
