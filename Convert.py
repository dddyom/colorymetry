import numpy as np


class ConvertHeader:

    @staticmethod
    def xyz(source_array):
        # print("source array\n", source_array[0][0:5])
        xyz_array = Conversion.rgb2xyz(source_array)
        # print("xyz\n", xyz_array[0][0:5])
        rgb_result_array = Conversion.xyz2rgb(xyz_array)
        # print("result\n", rgb_result_array[0][0:5])

        return rgb_result_array

    @staticmethod
    def cie_lab(source_array):
        # print("source array\n", source_array[0][0:5])
        xyz_array = Conversion.rgb2xyz(source_array)
        # print("xyz before lab\n", xyz_array[0][0:5])
        cie_lab_array = Conversion.xyz2cie_lab(xyz_array)
        # print("cie lab\n", lab_array[0][0:5])
        xyz_after_array = Conversion.cie_lab2xyz(cie_lab_array)
        # print("xyz after lab\n", xyz_after_array[0][0:5])
        rgb_result_array = Conversion.xyz2rgb(xyz_after_array)
        # print("result\n", rgb_result_array[0][0:5])
        return rgb_result_array

    @staticmethod
    def hunter_lab(source_array):
        # print("source array\n", source_array[0][0:5])
        xyz_array = Conversion.rgb2xyz(source_array)
        # print("xyz before lab\n", xyz_array[0][0:5])
        lab_array = Conversion.xyz2hunter_lab(xyz_array)
        # print("hunter lab\n", lab_array[0][0:5])
        xyz_after_array = Conversion.hunter_lab2xyz(lab_array)
        # print("xyz after lab\n", xyz_after_array[0][0:5])
        rgb_result_array = Conversion.xyz2rgb(xyz_after_array)
        # print("result\n", rgb_result_array[0][0:5])
        return rgb_result_array

    @staticmethod
    def hsv(source_array):
        # print("source array\n", source_array[0][0:5])
        hsv_array = Conversion.rgb2hsv(source_array)
        # print("hsv\n", hsv_array[0][0:5])
        rgb_result_array = Conversion.hsv2rgb(hsv_array)
        # print("result\n", rgb_result_array[0][0:5])
        return rgb_result_array

    @staticmethod
    def hsl(source_array):
        # print("source array\n", source_array[0][0:5])
        hsl_array = Conversion.rgb2hsl(source_array)
        # print("hsl\n", hsl_array[0][0:5])
        rgb_result_array = Conversion.hsl2rgb(hsl_array)
        # print("result\n", rgb_result_array[0][0:5])
        return rgb_result_array

    @staticmethod
    def luv(source_array):
        # print("source array\n", source_array[0][0:5])
        xyz_array = Conversion.rgb2xyz(source_array)
        # print("xyz before luv\n", xyz_array[0][0:5])
        luv_array = Conversion.xyz2luv(xyz_array)
        # print("luv\n", luv_array[0][0:5])
        xyz_after_array = Conversion.luv2xyz(luv_array)
        # print("xyz after luv\n", xyz_after_array[0][0:5])
        rgb_result_array = Conversion.xyz2rgb(xyz_after_array)
        # print("result\n", rgb_result_array[0][0:5])
        return rgb_result_array

    @staticmethod
    def cmy(source_array):  # ?
        # print("source array\n", source_array[0][0:5])
        cmy_array = Conversion.rgb2cmy(source_array)
        # print("cmy\n", cmy_array[0][0:5])
        rgb_result_array = Conversion.cmy2rgb(cmy_array)
        # print("result\n", rgb_result_array[0][0:5])
        return rgb_result_array

    @staticmethod
    def yxy(source_array):  # ?
        # print("source array\n", source_array[0][0:5])
        xyz_array = Conversion.rgb2xyz(source_array)
        # print("xyz before lab\n", xyz_array[0][0:5])
        yxy_array = Conversion.xyz2yxy(xyz_array)
        # print("yxy\n", yxy_array[0][0:5])
        xyz_after_array = Conversion.yxy2xyz(yxy_array)
        # print("xyz after yxy\n", xyz_after_array[0][0:5])
        rgb_result_array = Conversion.xyz2rgb(xyz_after_array)
        # print("result\n", rgb_result_array[0][0:5])
        return rgb_result_array


class Conversion:

    @staticmethod
    def rgb2xyz(rgb_array):
        xyz_array = []
        for rgb_line in rgb_array:
            xyz_line = []
            for rgb_pixel in rgb_line:

                # 0-255 (sRGB)
                sR = rgb_pixel[0]
                sG = rgb_pixel[1]
                sB = rgb_pixel[2]

                # normalize to 0-1 (RGB)
                R = sR / 255
                G = sG / 255
                B = sB / 255

                if R > 0.04045:  # гамма https://en.wikipedia.org/wiki/SRGB. В конце файла ещё
                    R = ((R + 0.055) / 1.055) ** 2.4
                else:
                    R = R / 12.92

                if G > 0.04045:
                    G = ((G + 0.055) / 1.055) ** 2.4
                else:
                    G = G / 12.92

                if B > 0.04045:
                    B = ((B + 0.055) / 1.055) ** 2.4
                else:
                    B = B / 12.92

                R = R * 100
                G = G * 100
                B = B * 100

                X = R * 0.412453 + G * 0.357580 + B * 0.180423  # перемножение матриц через констнты из CIE Rec.709
                Y = R * 0.212671 + G * 0.715160 + B * 0.072169
                Z = R * 0.019334 + G * 0.119193 + B * 0.950227

                xyz_pixel = np.array([X, Y, Z])
                xyz_line.append(xyz_pixel)
            xyz_array.append(np.array(xyz_line))
        return np.array(xyz_array)

    @staticmethod
    def xyz2rgb(xyz_array):
        rgb_array = []
        for xyz_line in xyz_array:
            rgb_line = []
            for xyz_pixel in xyz_line:

                X = xyz_pixel[0]
                Y = xyz_pixel[1]
                Z = xyz_pixel[2]

                # normalize
                var_X = X / 100
                var_Y = Y / 100
                var_Z = Z / 100

                var_R = var_X * 3.240479 + var_Y * -1.537156 + var_Z * -0.498536  # константы из CIE Rec. 709
                var_G = var_X * -0.969256 + var_Y * 1.875992 + var_Z * 0.0441556
                var_B = var_X * 0.055648 + var_Y * -0.204043 + var_Z * 1.057311

                if var_R > 0.0031308:  # https://ru.wikipedia.org/wiki/SRGB
                    var_R = 1.055 * (var_R ** (1 / 2.4)) - 0.055
                else:
                    var_R = 12.92 * var_R

                if var_G > 0.0031308:
                    var_G = 1.055 * (var_G ** (1 / 2.4)) - 0.055
                else:
                    var_G = 12.92 * var_G

                if var_B > 0.0031308:
                    var_B = 1.055 * (var_B ** (1 / 2.4)) - 0.055
                else:
                    var_B = 12.92 * var_B

                sR = var_R * 255
                sG = var_G * 255
                sB = var_B * 255

                rgb_pixel = np.array([sR, sG, sB])
                rgb_line.append(rgb_pixel)

            rgb_array.append(np.array(rgb_line))

        return np.array(rgb_array)

    @staticmethod
    def xyz2cie_lab(xyz_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        lab_array = []
        for xyz_line in xyz_array:
            lab_line = []
            for xyz_pixel in xyz_line:

                X = xyz_pixel[0]
                Y = xyz_pixel[1]
                Z = xyz_pixel[2]

                var_X = X / reference - X
                var_Y = Y / reference - Y
                var_Z = Z / reference - Z

                if var_X > 0.008856:
                    var_X = var_X ** (1 / 3)
                else:
                    var_X = (7.787 * var_X) + (16 / 116)

                if var_Y > 0.008856:
                    var_Y = var_Y ** (1 / 3)
                else:
                    var_Y = (7.787 * var_Y) + (16 / 116)

                if var_Z > 0.008856:
                    var_Z = var_Z ** (1 / 3)
                else:
                    var_Z = (7.787 * var_Z) + (16 / 116)

                CIE_L = (116 * var_Y) - 16
                CIE_a = 500 * (var_X - var_Y)
                CIE_b = 200 * (var_Y - var_Z)

                lab_pixel = np.array([CIE_L, CIE_a, CIE_b])
                lab_line.append(lab_pixel)

            lab_array.append(np.array(lab_line))

        return np.array(lab_array)

    @staticmethod
    def cie_lab2xyz(lab_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        xyz_array = []
        for lab_line in lab_array:
            xyz_line = []
            for lab_pixel in lab_line:

                CIE_L = lab_pixel[0]
                CIE_a = lab_pixel[1]
                CIE_b = lab_pixel[2]

                var_Y = (CIE_L + 16) / 116
                var_X = CIE_a / 500 + var_Y
                var_Z = var_Y - CIE_b / 200

                if var_Y ** 3 > 0.008856:
                    var_Y = var_Y ** 3
                else:
                    var_Y = (var_Y - 16 / 116) / 7.787

                if var_X ** 3 > 0.008856:
                    var_X = var_X ** 3
                else:
                    var_X = (var_X - 16 / 116) / 7.787

                if var_Z ** 3 > 0.008856:
                    var_Z = var_Z ** 3
                else:
                    var_Z = (var_Z - 16 / 116) / 7.787

                X = abs(var_X)
                Y = abs(var_Y)
                Z = abs(var_Z)

                xyz_pixel = np.array([X, Y, Z])
                xyz_line.append(xyz_pixel)

            xyz_array.append(np.array(xyz_line))

        return np.array(xyz_array)

    @staticmethod
    def xyz2hunter_lab(xyz_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        lab_array = []
        for xyz_line in xyz_array:
            lab_line = []
            for xyz_pixel in xyz_line:

                X = xyz_pixel[0]
                Y = xyz_pixel[1]
                Z = xyz_pixel[2]

                var_Ka = (175.0 / 198.04) * (reference + reference)
                var_Kb = (70.0 / 218.11) * (reference + reference)

                hunter_L = 100 * (Y / reference)**(1/2)
                hunter_a = var_Ka * (((X / reference) - (Y / reference)) / abs((Y / reference))**(1/2))
                hunter_b = var_Kb * (((Y / reference) - (Z / reference)) / abs((Y / reference))**(1/2))

                lab_pixel = np.array([hunter_L, hunter_a, hunter_b])
                lab_line.append(lab_pixel)

            lab_array.append(np.array(lab_line))

        return np.array(lab_array)

    @staticmethod
    def hunter_lab2xyz(lab_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        xyz_array = []
        for lab_line in lab_array:
            xyz_line = []
            for lab_pixel in lab_line:

                hunter_L = lab_pixel[0]
                hunter_a = lab_pixel[1]
                hunter_b = lab_pixel[2]

                var_Ka = (175.0 / 198.04) * (reference + reference)
                var_Kb = (70.0 / 218.11) * (reference + reference)

                Y = ((hunter_L / reference) ** 2) * 100.0
                X = (hunter_a / var_Ka * (Y / reference)**(1/2) + (Y / reference)) * reference
                Z = - (hunter_b / var_Kb * (Y / reference)**(1/2) - (Y / reference)) * reference

                xyz_pixel = np.array([X, Y, Z])
                xyz_line.append(xyz_pixel)

            xyz_array.append(np.array(xyz_line))

        return np.array(xyz_array)

    @staticmethod
    def rgb2hsv(rgb_array):
        hsv_array = []
        for rgb_line in rgb_array:
            hsv_line = []
            for rgb_pixel in rgb_line:

                # 0-255 (sRGB)
                R = rgb_pixel[0]
                G = rgb_pixel[1]
                B = rgb_pixel[2]

                # normalize to 0-1 (RGB)

                var_R = R / 255
                var_G = G / 255
                var_B = B / 255

                var_Min = min(var_R, var_G, var_B)
                var_Max = max(var_R, var_G, var_B)
                del_Max = var_Max - var_Min

                V = var_Max

                if del_Max == 0:
                    H = 0
                    S = 0
                else:
                    S = del_Max / var_Max

                    del_R = (((var_Max - var_R) / 6) + (del_Max / 2)) / del_Max
                    del_G = (((var_Max - var_G) / 6) + (del_Max / 2)) / del_Max
                    del_B = (((var_Max - var_B) / 6) + (del_Max / 2)) / del_Max

                    if var_R == var_Max:
                        H = del_B - del_G
                    elif var_G == var_Max:
                        H = (1 / 3) + del_R - del_B
                    elif var_B == var_Max:
                        H = (2 / 3) + del_G - del_R

                    if H < 0:
                        H = H + 1
                    if H > 1:
                        H = H - 1

                hsv_pixel = np.array([H, S, V])
                hsv_line.append(hsv_pixel)
            hsv_array.append(np.array(hsv_line))
        return np.array(hsv_array)

    @staticmethod
    def hsv2rgb(hsv_array):
        rgb_array = []
        for hsv_line in hsv_array:
            rgb_line = []
            for hsv_pixel in hsv_line:
                H = hsv_pixel[0]
                S = hsv_pixel[1]
                V = hsv_pixel[2]

                if S == 0:
                    R = V * 255
                    G = V * 255
                    B = V * 255
                else:
                    var_h = H * 6

                    if var_h == 6: var_h = 0

                    var_i = int(var_h)
                    var_1 = V * (1 - S)
                    var_2 = V * (1 - S * (var_h - var_i))
                    var_3 = V * (1 - S * (1 - (var_h - var_i)))

                    if var_i == 0:
                        var_r = V
                        var_g = var_3
                        var_b = var_1

                    elif var_i == 1:
                        var_r = var_2
                        var_g = V
                        var_b = var_1

                    elif var_i == 2:
                        var_r = var_1
                        var_g = V
                        var_b = var_3

                    elif var_i == 3:
                        var_r = var_1
                        var_g = var_2
                        var_b = V

                    elif var_i == 4:
                        var_r = var_3
                        var_g = var_1
                        var_b = V

                    else:
                        var_r = V
                        var_g = var_1
                        var_b = var_2

                    R = var_r * 255
                    G = var_g * 255
                    B = var_b * 255

                rgb_pixel = np.array([R, G, B])
                rgb_line.append(rgb_pixel)
            rgb_array.append(np.array(rgb_line))
        return np.array(rgb_array)

    @staticmethod
    def rgb2hsl(rgb_array):
        hsl_array = []
        for rgb_line in rgb_array:
            hsl_line = []
            for rgb_pixel in rgb_line:
                # 0-255 (sRGB)
                R = rgb_pixel[0]
                G = rgb_pixel[1]
                B = rgb_pixel[2]

                # normalize to 0-1 (RGB)

                var_R = R / 255
                var_G = G / 255
                var_B = B / 255

                var_Min = min(var_R, var_G, var_B)
                var_Max = max(var_R, var_G, var_B)
                del_Max = var_Max - var_Min

                L = (var_Max + var_Min) / 2

                if del_Max == 0:
                    H = 0
                    S = 0

                else:
                    if L < 0.5:
                        S = del_Max / (var_Max + var_Min)
                    else:
                        S = del_Max / (2 - var_Max - var_Min)

                    del_R = (((var_Max - var_R) / 6) + (del_Max / 2)) / del_Max
                    del_G = (((var_Max - var_G) / 6) + (del_Max / 2)) / del_Max
                    del_B = (((var_Max - var_B) / 6) + (del_Max / 2)) / del_Max

                    if var_R == var_Max:
                        H = del_B - del_G
                    elif var_G == var_Max:
                        H = (1 / 3) + del_R - del_B
                    elif var_B == var_Max:
                        H = (2 / 3) + del_G - del_R

                    if H < 0: H += 1
                    if H > 1: H -= 1

                hsl_pixel = np.array([H, S, L])
                hsl_line.append(hsl_pixel)
            hsl_array.append(np.array(hsl_line))
        return np.array(hsl_array)

    @staticmethod
    def hsl2rgb(hsl_array):
        rgb_array = []
        for hsl_line in hsl_array:
            rgb_line = []
            for hsl_pixel in hsl_line:
                H = hsl_pixel[0]
                S = hsl_pixel[1]
                L = hsl_pixel[2]

                if S == 0:
                    R = L * 255
                    G = L * 255
                    B = L * 255
                else:
                    if L < 0.5:
                        var_2 = L * (1 + S)
                    else:
                        var_2 = (L + S) - (S * L)

                    var_1 = 2 * L - var_2

                    R = 255 * Conversion.hue2rgb(var_1, var_2, H + (1 / 3))
                    G = 255 * Conversion.hue2rgb(var_1, var_2, H)
                    B = 255 * Conversion.hue2rgb(var_1, var_2, H - (1 / 3))

                rgb_pixel = np.array([R, G, B])
                rgb_line.append(rgb_pixel)
            rgb_array.append(np.array(rgb_line))
        return np.array(rgb_array)

    @staticmethod
    def hue2rgb(v1, v2, vH):
        """For hsl conversion"""
        if vH < 0:
            vH += 1
        if vH > 1:
            vH -= 1
        if (6 * vH) < 1:
            return v1 + (v2 - v1) * 6 * vH
        if (2 * vH) < 1:
            return v2
        if (3 * vH) < 2:
            return v1 + (v2 - v1) * ((2 / 3) - vH) * 6
        return v1

    @staticmethod
    def xyz2luv(xyz_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        luv_array = []
        for xyz_line in xyz_array:
            luv_line = []
            for xyz_pixel in xyz_line:

                X = xyz_pixel[0]
                Y = xyz_pixel[1]
                Z = xyz_pixel[2]

                var_U = (4 * X) / (X + (15 * Y) + (3 * Z))
                var_V = (9 * Y) / (X + (15 * Y) + (3 * Z))

                var_Y = Y / 100
                if var_Y > 0.008856:
                    var_Y = var_Y ^ (1 / 3)
                else:
                    var_Y = (7.787 * var_Y) + (16 / 116)

                ref_U = (4 * reference - X) / (reference - X + (15 * reference - Y) + (3 * reference - Z))
                ref_V = (9 * reference - Y) / (reference - X + (15 * reference - Y) + (3 * reference - Z))

                CIE_L = (116 * var_Y) - 16
                CIE_u = 13 * CIE_L * (var_U - ref_U)
                CIE_v = 13 * CIE_L * (var_V - ref_V)

                luv_pixel = np.array([CIE_L, CIE_u, CIE_v])
                luv_line.append(luv_pixel)

            luv_array.append(np.array(luv_line))

        return np.array(luv_array)

    @staticmethod
    def luv2xyz(luv_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        xyz_array = []
        for luv_line in luv_array:
            xyz_line = []
            for luv_pixel in luv_line:

                CIE_L = luv_pixel[0]
                CIE_u = luv_pixel[1]
                CIE_v = luv_pixel[2]

                var_Y = (CIE_L+ 16) / 116
                if var_Y ** 3 > 0.008856:
                    var_Y = var_Y ** 3
                else:
                    var_Y = ( var_Y - 16 / 116 ) / 7.787

                ref_U = (4 * reference - X) / (reference - X + (15 * reference - Y) + (3 * reference - Z))
                ref_V = (9 * reference - Y) / (reference - X + (15 * reference - Y) + (3 * reference - Z))

                var_U = CIE_u / (13 * CIE_L) + ref_U
                var_V = CIE_v / (13 * CIE_L) + ref_V

                Y = var_Y * 100
                X = - (9 * Y * var_U) / ((var_U - 4) * var_V - var_U * var_V)
                Z = (9 * Y - (15 * var_V * Y) - (var_V * X)) / (3 * var_V)
                xyz_pixel = np.array([X, Y, Z])
                xyz_line.append(xyz_pixel)

            xyz_array.append(np.array(xyz_line))

        return np.array(xyz_array)

    @staticmethod
    def rgb2cmy(rgb_array):
        cmy_array = []
        for rgb_line in rgb_array:
            cmy_line = []
            for rgb_pixel in rgb_line:

                # 0-255 (sRGB)
                sR = rgb_pixel[0]
                sG = rgb_pixel[1]
                sB = rgb_pixel[2]

                # normalize to 0-1 (RGB)
                R = sR / 255
                G = sG / 255
                B = sB / 255


                C = 1 - R
                M = 1 - G
                Y = 1 - B

                cmy_pixel = np.array([C, M, Y])
                cmy_line.append(cmy_pixel)
            cmy_array.append(np.array(cmy_line))
        return np.array(cmy_array)

    @staticmethod
    def cmy2rgb(cmy_array):
        rgb_array = []
        for cmy_line in cmy_array:
            rgb_line = []
            for cmy_pixel in cmy_line:

                C = cmy_pixel[0]
                M = cmy_pixel[1]
                Y = cmy_pixel[2]

                R = (1 - C) * 255
                G = (1 - M) * 255
                B = (1 - Y) * 255

                rgb_pixel = np.array([R, G, B])
                rgb_line.append(rgb_pixel)

            rgb_array.append(np.array(rgb_line))

        return np.array(rgb_array)

    @staticmethod
    def xyz2yxy(xyz_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        yxy_array = []
        for xyz_line in xyz_array:
            yxy_line = []
            for xyz_pixel in xyz_line:

                X = xyz_pixel[0]
                Y = xyz_pixel[1]
                Z = xyz_pixel[2]

                var_Y = Y
                var_x = X / (X + Y + Z)
                var_y = Y / (X + Y + Z)

                yxy_pixel = np.array([var_Y, var_x, var_y])
                yxy_line.append(yxy_pixel)

            yxy_array.append(np.array(yxy_line))

        return np.array(yxy_array)

    @staticmethod
    def yxy2xyz(yxy_array):
        reference = 100  # Reference from cie 1964 equal energy (100)
        xyz_array = []
        for yxy_line in yxy_array:
            xyz_line = []
            for yxy_pixel in yxy_line:

                var_Y = yxy_pixel[0]
                var_x = yxy_pixel[1]
                var_y = yxy_pixel[2]

                X = var_x * (var_Y / var_y)
                Y = var_Y
                Z = (1 - var_x - var_y) * (var_Y / var_y)

                xyz_pixel = np.array([X, Y, Z])
                xyz_line.append(xyz_pixel)

            xyz_array.append(np.array(xyz_line))

        return np.array(xyz_array)
