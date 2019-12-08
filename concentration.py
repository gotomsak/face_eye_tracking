max_hindo = 4.6
min_hindo = 0.0
hindo = [0.0, 0.0, 0.0, 0.6, 1.0]



concentration_list = []
# (「瞬き回数or頻度」- 最大瞬き頻度) / (最低瞬き頻度 - 最大瞬き頻度)
for i in hindo:
    concentration_list.append(round((i - max_hindo)/(min_hindo - max_hindo),2))

print(concentration_list)