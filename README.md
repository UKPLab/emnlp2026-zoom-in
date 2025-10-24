## Test Run
activate the venv with the custom trl. Open a screen and set `CUDA_VISIBLE_DEVICES=0`. Then call 
```
trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --limit_image_per_prompt 3 --max_pixels 784000
```
Detach the screen, open a new screen and set `CUDA_VISIBLE_DEVICES=1`. Then call 
```
cd src/scripts
bash run_training_tool_test.sh
```

## Train Run
```
trl vllm-serve --model TIGER-Lab/PixelReasoner-WarmStart --limit_image_per_prompt 3 --max_pixels 3920000 --min_pixels 196000 --enable_prefix_caching True
```
```
trl vllm-serve --model Qwen/Qwen2.5-VL-7B-Instruct --limit_image_per_prompt 3 --max_pixels 3920000 --min_pixels 196000 --enable_prefix_caching True
```


max img tokens | generations | iterations | per device bs | num gpus | grad acc | global bs          | steps
1000           |         8   |        2   | 8             | 7        | 5        |      280=8x7x5     |  1/17.5 = 8x2 / 8x7x5
                                                            6          6        |      288=8x6x6     |  1/18   = 8x2 / 8x6x6
2000                                        4               6          12              288=4x6x12    |  1/18
                         7                  4               7          10              280              1/20   = 7x2 / 4x7x10 
                         6                  4               6          12              288              1/24   = 6x2 / 4x6x12
4000                     8                  2               4          35              280              1/17.5
                         7                  2               7          20              280              1/20   = 7x2 / 2x7x20
                         6                  2               6          24              288              1/24   = 6x2 / 2x6x24


clip-value gamma          |  threshold tau         |  scaling factor N        | reward weight w | reward bound
1.5                          0                        256                       0.06              -0.09, +0.09
                             0.1                                                0.07              -0.112, +0.098
                             0.5                                                0.1               -0.2  , +0.1
                             0.3                                                0.08              -0.144, +0.096