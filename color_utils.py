from utils.color_recognition.src.color_classification_image import get_color, get_rgb
import cv2
import numpy as np
import webcolors

def get_pointing_color(image, hand_side, index_tip, crop_size=(15, 15), shift_factor=0):
    """
    Extracts the color from a small cropped area near the index fingertip,
    adjusting the crop position to avoid fingers in the selection.

    Parameters:
        image (numpy array): Input image.
        hand_side (str): 'right' or 'left' hand.
        index_tip (tuple): (x, y) coordinates of the index fingertip.
        crop_size (tuple): (width, height) of the cropped region.
        shift_factor (int): How much to shift the crop region to avoid fingers.

    Returns:
        color (tuple): Average color of the cropped area.
    """
    h, w, _ = image.shape
    crop_w, crop_h = crop_size

    crop_area = None

    if hand_side == 'right':
        # Shift more to the top-left
        right_x, right_y = index_tip
        right_x1 = max(0, right_x - crop_w - shift_factor)  # Shift left
        right_y1 = max(0, right_y - crop_h - shift_factor)  # Shift up
        right_x2 = min(w, right_x - shift_factor)  # Keep inside bounds
        right_y2 = min(h, right_y - shift_factor)

        if right_y2 > right_y1 and right_x2 > right_x1:
            crop_area = image[right_y1:right_y2, right_x1:right_x2]
    else:  # Left hand
        # Shift more to the top-right
        left_x, left_y = index_tip
        left_x1 = max(0, left_x + shift_factor)  # Shift right
        left_y1 = max(0, left_y - crop_h - shift_factor)  # Shift up
        left_x2 = min(w, left_x + crop_w + shift_factor)  # Keep inside bounds
        left_y2 = min(h, left_y - shift_factor)
    
        if left_y2 > left_y1 and left_x2 > left_x1:
            crop_area = image[left_y1:left_y2, left_x1:left_x2]

    if crop_area is None or crop_area.size == 0:
        print("Warning: Crop area is empty or invalid.")
        return "No color identified"

    # color = get_color(crop_area)
    color = get_color_from_crop(crop_area)

    # cv2.imwrite("crop_area.jpg", crop_area)

    # enhanced_crop = adjust_brightness_contrast(crop_area)


    # rgb = get_rgb(crop_area)
    # color  = closest_color(tuple(rgb))
    
    return color


def closest_color(requested_colour):
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    name = min_colours[min(min_colours.keys())]


    print(f"color name {name}\n"*100)
    return formatted_colors[name]

def get_color_from_crop(crop_area):
    """
    Computes the average color of a cropped image region and returns its closest color name.
    
    Parameters:
        crop_area (numpy array): Cropped image region.

    Returns:
        str: Name of the closest color.
    """
    if crop_area is None or crop_area.size == 0:
        return "Invalid crop area"
    
    # Compute mean for each color channel (BGR in OpenCV)
    avg_color = np.mean(crop_area, axis=(0, 1))  # Average over height & width
    avg_color = tuple(map(int, avg_color[::-1]))  # Convert BGR to RGB format

    return closest_color(avg_color)

def adjust_brightness_contrast(image, alpha=1.2, beta=0):
    """
    Adjusts the brightness and contrast of an image.

    Parameters:
        image (numpy array): Input cropped image.
        alpha (float): Contrast control (1.0-3.0).
        beta (int): Brightness control (0-100).

    Returns:
        numpy array: Adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

formatted_colors = {
    'aliceblue': 'Blue',
    'antiquewhite': 'White',
    'aqua': 'Cyan',
    'aquamarine': 'Green',
    'azure': 'Blue',
    'beige': 'Cream',
    'bisque': 'Peach',
    'black': 'Black',
    'blanchedalmond': 'Cream',
    'blue': 'Blue',
    'blueviolet': 'Purple',
    'brown': 'Brown',
    'burlywood': 'burlywood',
    'cadetblue': 'Blue',
    'chartreuse': 'Green',
    'chocolate': 'Brown',
    'coral': 'Pink',
    'cornflowerblue': 'Blue',
    'cornsilk': 'Yellow',
    'crimson': 'Red',
    'cyan': 'Cyan',
    'darkblue': 'Blue',
    'darkcyan': 'Blue',
    'darkgoldenrod': 'Gold',
    'darkgray': 'Gray',
    'darkgrey': 'Gray',
    'darkgreen': 'Forest Green',
    'darkkhaki': 'Green',
    'darkmagenta': 'Dark Purple',
    'darkolivegreen': 'Green',
    'darkorange': 'Orange',
    'darkorchid': 'Purple',
    'darkred': 'Red',
    'darksalmon': 'orange',
    'darkseagreen': 'Green',
    'darkslateblue': 'Dark Blue',
    'darkslategray': 'Dark Gray',
    'darkslategrey': 'Dark Gray',
    'darkturquoise': 'green',
    'darkviolet': 'Purple',
    'deeppink': 'Red',
    'deepskyblue': 'Blue',
    'dimgray': 'Black',
    'dimgrey': 'Black',
    'dodgerblue': 'Blue',
    'firebrick': 'Red',
    'floralwhite': 'White',
    'forestgreen': 'Green',
    'fuchsia': 'Red',
    'gainsboro': 'Gray',
    'ghostwhite': 'white',
    'gold': 'Yellow',
    'goldenrod': 'Yellow',
    'gray': 'White',
    'grey': 'White',
    'green': 'Green',
    'greenyellow': 'Lime',
    'honeydew': 'Green',
    'hotpink': 'Red',
    'indianred': 'Red',
    'indigo': 'Blue',
    'ivory': 'White',
    'khaki': 'White',
    'lavender': 'White',
    'lavenderblush': 'White',
    'lawngreen': 'Green',
    'lemonchiffon': 'Yellow',
    'lightblue': 'Blue',
    'lightcoral': 'Red',
    'lightcyan': 'Blue',
    'lightgoldenrodyellow': 'Yellow',
    'lightgray': 'White',
    'lightgrey': 'White',
    'lightgreen': 'Green',
    'lightpink': 'Pink',
    'lightsalmon': 'Orange',
    'lightseagreen': 'Green',
    'lightskyblue': 'Blue',
    'lightslategray': 'Gray',
    'lightslategrey': 'Gray',
    'lightsteelblue': 'Blue',
    'lightyellow': 'Yellow',
    'lime': 'Green',
    'limegreen': 'Green',
    'linen': 'Beige',
    'magenta': 'Pink',
    'maroon': 'Red',
    'mediumaquamarine': 'Green',
    'mediumblue': 'Blue',
    'mediumorchid': 'Purple',
    'mediumpurple': 'Purple',
    'mediumseagreen': 'Green',
    'mediumslateblue': 'Blue',
    'mediumspringgreen': 'Green',
    'mediumturquoise': 'Green',
    'mediumvioletred': 'Pink',
    'midnightblue': 'Blue',
    'mintcream': 'Green',
    'mistyrose': 'Pink',
    'moccasin': 'Pink',
    'navajowhite': 'Pink',
    'navy': 'Blue',
    'oldlace': 'Cream',
    'olive': 'Green',
    'olivedrab': 'Green',
    'orange': 'Orange',
    'orangered': 'Orange',
    'orchid': 'Purple',
    'palegoldenrod': 'Yellow',
    'palegreen': 'Green',
    'paleturquoise': 'Green',
    'palevioletred': 'Pink',
    'papayawhip': 'White',
    'peachpuff': 'Pink',
    'peru': 'Orange',
    'pink': 'Pink',
    'plum': 'Purple',
    'powderblue': 'Blue',
    'purple': 'Purple',
    'red': 'Red',
    'rosybrown': 'Purple',
    'royalblue': 'Blue',
    'saddlebrown': 'Brown',
    'salmon': 'Orange',
    'sandybrown': 'Brown',
    'seagreen': 'Green',
    'seashell': 'White',
    'sienna': 'Brown',
    'silver': 'White',
    'skyblue': 'Blue',
    'slateblue': 'Blue',
    'slategray': 'Gray',
    'slategrey': 'Gray',
    'snow': 'White',
    'springgreen': 'Green',
    'steelblue': 'Blue',
    'tan': 'White',
    'teal': 'Teal',
    'thistle': 'Purple',
    'tomato': 'Red',
    'turquoise': 'Green',
    'violet': 'Purple',
    'wheat': 'Beige',
    'white': 'White',
    'whitesmoke': 'White',
    'yellow': 'Yellow',
    'yellowgreen': 'Yellow',
}


# formatted_colors = {
#     'aliceblue': 'Blue',
#     'antiquewhite': 'White',
#     'aqua': 'Aqua',
#     'aquamarine': 'Aquamarine',
#     'azure': 'Azure',
#     'beige': 'Beige',
#     'bisque': 'Bisque',
#     'black': 'Black',
#     'blanchedalmond': 'Blanched Almond',
#     'blue': 'Blue',
#     'blueviolet': 'Blue Violet',
#     'brown': 'Brown',
#     'burlywood': 'Burly Wood',
#     'cadetblue': 'Cadet Blue',
#     'chartreuse': 'Chartreuse',
#     'chocolate': 'Chocolate',
#     'coral': 'Coral',
#     'cornflowerblue': 'Cornflower Blue',
#     'cornsilk': 'Cornsilk',
#     'crimson': 'Crimson',
#     'cyan': 'Cyan',
#     'darkblue': 'Dark Blue',
#     'darkcyan': 'Dark Cyan',
#     'darkgoldenrod': 'Dark Goldenrod',
#     'darkgray': 'Black',
#     'darkgrey': 'Black',
#     'darkgreen': 'Dark Green',
#     'darkkhaki': 'Dark Khaki',
#     'darkmagenta': 'Dark Magenta',
#     'darkolivegreen': 'Dark Olive Green',
#     'darkorange': 'Dark Orange',
#     'darkorchid': 'Dark Orchid',
#     'darkred': 'Dark Red',
#     'darksalmon': 'Red',
#     'darkseagreen': 'Sea Green',
#     'darkslateblue': 'Blue',
#     'darkslategray': 'Gray',
#     'darkslategrey': 'Grey',
#     'darkturquoise': 'Dark Turquoise',
#     'darkviolet': 'Dark Violet',
#     'deeppink': 'Deep Pink',
#     'deepskyblue': 'Deep Sky Blue',
#     'dimgray': 'Dim Gray',
#     'dimgrey': 'Dim Grey',
#     'dodgerblue': 'Dodger Blue',
#     'firebrick': 'Fire Brick',
#     'floralwhite': 'Floral White',
#     'forestgreen': 'Forest Green',
#     'fuchsia': 'Fuchsia',
#     'gainsboro': 'Grey',
#     'ghostwhite': 'Ghost White',
#     'gold': 'Gold',
#     'goldenrod': 'Goldenrod',
#     'gray': 'Gray',
#     'grey': 'Grey',
#     'green': 'Green',
#     'greenyellow': 'Green Yellow',
#     'honeydew': 'Honeydew',
#     'hotpink': 'Hot Pink',
#     'indianred': 'Indian Red',
#     'indigo': 'Indigo',
#     'ivory': 'Ivory',
#     'khaki': 'Khaki',
#     'lavender': 'Lavender',
#     'lavenderblush': 'Lavender Blush',
#     'lawngreen': 'Lawn Green',
#     'lemonchiffon': 'Lemon Chiffon',
#     'lightblue': 'Blue',
#     'lightcoral': 'Orange',
#     'lightcyan': 'Cyan',
#     'lightgoldenrodyellow': 'Yellow',
#     'lightgray': 'Gray',
#     'lightgrey': 'Grey',
#     'lightgreen': 'Green',
#     'lightpink': 'Pink',
#     'lightsalmon': 'Salmon',
#     'lightseagreen': 'Sea Green',
#     'lightskyblue': 'Sky Blue',
#     'lightslategray': 'Gray',
#     'lightslategrey': 'Grey',
#     'lightsteelblue': 'Light Steel Blue',
#     'lightyellow': 'Light Yellow',
#     'lime': 'Lime',
#     'limegreen': 'Lime Green',
#     'linen': 'Linen',
#     'magenta': 'Magenta',
#     'maroon': 'Maroon',
#     'mediumaquamarine': 'Medium Aquamarine',
#     'mediumblue': 'Medium Blue',
#     'mediumorchid': 'Medium Orchid',
#     'mediumpurple': 'Medium Purple',
#     'mediumseagreen': 'Medium Sea Green',
#     'mediumslateblue': 'Medium Slate Blue',
#     'mediumspringgreen': 'Medium Spring Green',
#     'mediumturquoise': 'Medium Turquoise',
#     'mediumvioletred': 'Medium Violet Red',
#     'midnightblue': 'Midnight Blue',
#     'mintcream': 'Mint Cream',
#     'mistyrose': 'Misty Rose',
#     'moccasin': 'Moccasin',
#     'navajowhite': 'Navajo White',
#     'navy': 'Navy',
#     'oldlace': 'Old Lace',
#     'olive': 'Olive',
#     'olivedrab': 'Olive Drab',
#     'orange': 'Orange',
#     'orangered': 'Orange Red',
#     'orchid': 'Orchid',
#     'palegoldenrod': 'Pale Goldenrod',
#     'palegreen': 'Pale Green',
#     'paleturquoise': 'Pale Turquoise',
#     'palevioletred': 'Pale Violet Red',
#     'papayawhip': 'Papaya Whip',
#     'peachpuff': 'Peach Puff',
#     'peru': 'Peru',
#     'pink': 'Pink',
#     'plum': 'Plum',
#     'powderblue': 'Powder Blue',
#     'purple': 'Purple',
#     'red': 'Red',
#     'rosybrown': 'Brown',
#     'royalblue': 'Royal Blue',
#     'saddlebrown': 'Saddle Brown',
#     'salmon': 'Salmon',
#     'sandybrown': 'Sandy Brown',
#     'seagreen': 'Sea Green',
#     'seashell': 'Seashell',
#     'sienna': 'Sienna',
#     'silver': 'Silver',
#     'skyblue': 'Sky Blue',
#     'slateblue': 'Slate Blue',
#     'slategray': 'Gray',
#     'slategrey': 'Grey',
#     'snow': 'White',
#     'springgreen': 'Spring Green',
#     'steelblue': 'Steel Blue',
#     'tan': 'Tan',
#     'teal': 'Teal',
#     'thistle': 'Thistle',
#     'tomato': 'Red',
#     'turquoise': 'Turquoise',
#     'violet': 'Violet',
#     'wheat': 'Wheat',
#     'white': 'White',
#     'whitesmoke': 'White Smoke',
#     'yellow': 'Yellow',
#     'yellowgreen': 'Yellow Green',
# }


# Example usage
if __name__ == "__main__":
    # image = cv2.imread("hand_image.jpg")  # Load an image
    image = cv2.imread('/Users/rueichechang/Projects/worldscribe/frame.png')
    # image = cv2.imread('/Users/rueichechang/Projects/worldscribe/cropped_object_0.png')

    right_index_tip = (200, 300)  # Example right index fingertip location
    left_index_tip = (100, 300)   # Example left index fingertip location

    # get_color(image)
    
    color = get_pointing_color(image, 'right', right_index_tip)
    print(color)