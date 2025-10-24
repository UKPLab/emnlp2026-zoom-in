import numpy as np



def reduce_img_per_sample(img_per_sample: list, masked_image_ids: list) -> list:
    img_per_sample = np.array(img_per_sample, dtype=int)
    masked_image_ids = np.array(masked_image_ids, dtype=int)

    cumulative_img_per_sample = np.cumsum(img_per_sample)

    subtract_img_per_sample = np.zeros(len(img_per_sample), dtype=int)

    for mi_idx in masked_image_ids:
        for cum_idx in range(len(cumulative_img_per_sample)):
            if cumulative_img_per_sample[cum_idx] > mi_idx and (cum_idx == 0 or cumulative_img_per_sample[cum_idx - 1] <= mi_idx):
                subtract_img_per_sample[cum_idx] += 1
                break

    reduced_img_per_sample = img_per_sample - subtract_img_per_sample

    return reduced_img_per_sample.tolist()

if __name__ == "__main__":
    img_per_sample = [2, 1, 2, 3, 1]
    masked_image_ids = [0, 4, 6, 8]

    print(reduce_img_per_sample(img_per_sample, masked_image_ids))


