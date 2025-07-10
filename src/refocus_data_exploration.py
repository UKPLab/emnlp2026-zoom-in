from PIL import Image, ImageDraw
from typing import List


def transform(bounding_boxes: List[dict], image_path="str", output_path="str"):
    with Image.open(image_path) as im:

        draw = ImageDraw.Draw(im)
        for idx, bounding_box in enumerate(bounding_boxes):

            if bounding_box["transform"] == "highlight":
                draw.rectangle((bounding_box["x0"], bounding_box["y0"],
                                bounding_box["x1"], bounding_box["y1"]),
                               outline=bounding_box["color"], width=5)

            elif bounding_box["transform"] == "mask":
                draw.rectangle((bounding_box["x0"], bounding_box["y0"],
                                bounding_box["x1"], bounding_box["y1"])
                               , fill=bounding_box["color"])

        # write to stdout
        im.save(output_path, "PNG")

if __name__ == "__main__":
    transform([{"x0": 0, "y0": 0, "x1": 500, "y1": 600, "color": "red", "transform": "mask"},
               {"x0": 0, "y0": 300, "x1": 1100, "y1": 500, "color": "green", "transform": "highlight"}
          ],
         image_path="../data/test_images/ukp_logo.png",
         output_path="../data/test_images/ukp_bb.png")