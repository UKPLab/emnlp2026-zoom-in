from qwen_vl_utils import smart_resize

QWEN2_5_MIN_PIXELS = 4*28*28
QWEN2_5_MAX_PIXELS = 16*1024*28*28

def get_resized_image_scales(height:int, width:int, min_pixels:int=None, max_pixels:int=None):

    if min_pixels is None:
        min_pixels = QWEN2_5_MIN_PIXELS
    if max_pixels is None:
        max_pixels = QWEN2_5_MAX_PIXELS


    resized_height, resized_width = smart_resize(height=height,
                                                 width=width,
                                                 min_pixels=min_pixels,
                                                 max_pixels=max_pixels)

    return resized_height, resized_width
