import torch
import numpy as np
import matplotlib.pyplot as plt

A = torch.tensor(-35.0)
B = torch.tensor(-42.0)

nom = torch.logaddexp(A, A)
denom = torch.logaddexp(A, B)

print(nom)
print(denom)
diff = nom - denom

print(f"diff: {diff}")
print(f"upper bound: {torch.log(torch.tensor(2.0))}")

# 2026-01-22 10:41:38
answer_scores_full = np.array([-1.3351e-04,  0.0000e+00, -1.0729e-06,  0.0000e+00,  0.0000e+00,
-1.1921e-07, -5.2452e-06, -2.3842e-07,  0.0000e+00,  0.0000e+00,
-4.7684e-07,  0.0000e+00, -1.7881e-06, -1.1921e-06,  0.0000e+00,
 0.0000e+00,  0.0000e+00, -2.6226e-06, -3.0994e-06,

                               -1.0859e+00,
-2.7188e+00, -2.5469e+00, -1.5381e-02, -4.3106e-04, -9.4531e-01,
-2.2656e+00, -2.2188e+00, -1.5198e-02, -5.9605e-07, -3.1875e+00,
-9.1406e-01, -2.0312e+00, -2.4219e+00,  0.0000e+00, -3.5763e-07,
-3.6133e-01, -2.0000e+00, -2.3750e+00, -4.7684e-07,  0.0000e+00,
-2.8610e-06, -3.5763e-06,  0.0000e+00, -5.8746e-04, -6.1951e-03,
-5.3883e-05, -1.1921e-07])

answer_scores_short = np.array([-1.3351e-04,  0.0000e+00, -1.0729e-06,  0.0000e+00,  0.0000e+00,
-1.1921e-07, -5.2452e-06, -2.3842e-07,  0.0000e+00,  0.0000e+00,
-4.7684e-07,  0.0000e+00, -1.7881e-06, -1.1921e-06,  0.0000e+00,
 0.0000e+00,  0.0000e+00, -2.6226e-06, -3.0994e-06,

                                -2.7188e+00,
-2.3906e+00, -2.3906e+00, -1.6689e-06, -1.3161e-04, -8.6328e-01,
-1.5234e+00, -2.5469e+00, -5.5000e+00, -4.7461e-01, -7.1526e-06,
-1.4531e+00, -1.8828e+00, -2.0312e+00, -2.4219e+00,  0.0000e+00,
-2.3842e-07, -2.5879e-02, -1.7734e+00, -2.6562e+00, -2.3750e+00,
-3.4571e-06, -2.3842e-07, -1.4067e-05, -3.6955e-06,  0.0000e+00,
-8.5449e-04, -1.8677e-02, -8.5354e-05,  0.0000e+00])

sm = np.sum(answer_scores_short[:len(answer_scores_full)] - answer_scores_full)
print(sm)
clip_sum = np.sum(np.clip(answer_scores_short[:len(answer_scores_full)] - answer_scores_full, -1.5, 1.5))
print(clip_sum)

#print(answer_scores_full - answer_scores_short)
plt.scatter(range(len(answer_scores_full)), answer_scores_full, color = "red")
plt.scatter(range(len(answer_scores_short)), answer_scores_short, color = "blue")
plt.show()
plt.scatter(range(len(answer_scores_full)), np.cumsum(answer_scores_full), color = "red")
plt.scatter(range(len(answer_scores_short)), np.cumsum(answer_scores_short), color = "blue")
plt.show()
plt.scatter(range(len(answer_scores_full)), answer_scores_short[:len(answer_scores_full)] - answer_scores_full, color = "blue")
plt.show()