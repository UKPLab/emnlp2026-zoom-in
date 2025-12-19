from open_r1.utils.utils import generate_bbox_2d_new_close_iou_targeted
import time
import matplotlib.pyplot as plt
from matplotlib import patches

if __name__ == "__main__":
    bbox_2d = (200, 300, 400, 500)
    plt.Rectangle((bbox_2d[0], bbox_2d[1]), bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1],)
    #plt.scatter(bbox_2d[0], bbox_2d[1], color = "red")
    #plt.scatter(bbox_2d[2], bbox_2d[3], color = "red")
    for _ in range(5):
        t0 = time.time()
        res = generate_bbox_2d_new_close_iou_targeted(bbox_2d=(200, 300, 400, 500),
                                          target_iou=0.1,
                                          image_size=(1000, 1000),
                                          #verify=True,
                                          return_debug=True,
                                          seed=None,
                                          tol = 0.1,
                                          max_tries=100
                                                      )
        t1 = time.time()
                                          #verify_max_tries=10000)
        print(f"bbox: {res[0]}")
        print(f"attempt: {res[1]["attempt"]}")
        print(f"error: {res[1]["abs_error"]}")
        print(f"{t1-t0} seconds")
    plt.show()