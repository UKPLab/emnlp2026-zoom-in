def zoom(img_x, img_y, bbox_2d, padding=(0.1, 0.1)):
    """
    Crop the image based on the bounding box coordinates.
    """
    padding_tr = (600.0 / img_x, 600.0 / img_y)
    padding = (min(padding[0], padding_tr[0]), min(padding[1], padding_tr[1]))

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0]) - padding[0],
                              float(bbox_2d[1]) - padding[1],
                              float(bbox_2d[2]) + padding[0],
                              float(bbox_2d[3]) + padding[1])
    else:
        normalized_bbox_2d = (float(bbox_2d[0]) / img_x - padding[0],
                              float(bbox_2d[1]) / img_y - padding[1],
                              float(bbox_2d[2]) / img_x + padding[0],
                              float(bbox_2d[3]) / img_y + padding[1])
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 = min(max(0, normalized_x1), 1)
    normalized_y1 = min(max(0, normalized_y1), 1)
    normalized_x2 = min(max(0, normalized_x2), 1)
    normalized_y2 = min(max(0, normalized_y2), 1)

    print(f"Zoom: Crop from ({int(normalized_x1 * img_x)},{int(normalized_y1 * img_y)}) to ({int(normalized_x2 * img_x)},{int(normalized_y2 * img_y)})")

def crop(img_x, img_y, bbox_2d,  padding=0.1):
    """
    Crop the image based on the bounding box coordinates.
    """
    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding,
                              float(bbox_2d[1])-padding,
                              float(bbox_2d[2])+padding,
                              float(bbox_2d[3])+padding)
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding,
                              float(bbox_2d[1])/img_y-padding,
                              float(bbox_2d[2])/img_x+padding,
                              float(bbox_2d[3])/img_y+padding)
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)

    print(
        f"Crop Normalized: Crop from ({normalized_x1*img_x},{normalized_y1*img_y}) to ({normalized_x2*img_x},{normalized_y2*img_y})")

if __name__ == "__main__":
    img_x = 6100
    img_y = 6100
    bbox_2d = (0.2, 0.2, 0.8, 0.8)
    crop(img_x, img_y, bbox_2d)
    zoom(img_x, img_y, bbox_2d)