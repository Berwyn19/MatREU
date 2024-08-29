with open('test_idx.txt', 'w') as file:
    NUM_TRAINING = 101
    for i in range(NUM_TRAINING):
        file.write(f"{i}\n")
# import numpy as np
# import random

# with open('tune_idx.txt', 'w') as file:
#     idxs = []
#     count = 0
#     NUM_TRAINING = 100
#     while count < NUM_TRAINING:
#         random_idx = random.randint(0, 3455)
#         if random_idx not in idxs:
#             file.write(f"{random_idx}\n")
#             idxs.append(random_idx)
#             count += 1
