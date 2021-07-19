import numpy as np
max_hindo = 16
min_hindo = 1
hindo = [[5053, 3971], [4988, 3653], [8859, 4865], [5353, 3984], [4336, 3275]]

concentration_list = []
# (「瞬き回数or頻度」- 最大瞬き頻度) / (最低瞬き頻度 - 最大瞬き頻度)
if np.array(hindo).ndim == 2:
    for i in range(len(hindo)):
        hindo[i] = sum(hindo[i])

for i in hindo:
    concentration_list.append(round((i - max(hindo))/(min(hindo) - max(hindo)),2))

print(concentration_list)