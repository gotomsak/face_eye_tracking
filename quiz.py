import random


# ランダムに4桁の整数を返す
def create_number(max, min):
    random.seed()
    return random.randint(min, max)


def random_bool():
    random.seed()
    return random.randint(0, 1)


# 足し算の結果を返す
def result_plus(n1, n2):
    return n1 + n2


def result_minus(n1, n2):
    return n1 - n2


# 足し算の結果の誤差+-100までを4つランダムに生成
def create_number_mistake(result):
    random.seed()
    mistake_list = []
    for i in range(3):
        mistake_list.append(random.randint(result - 50, result + 50))
    return mistake_list


if __name__ == '__main__':

    n1 = create_number(9999, 1000)
    n2 = create_number(9999, 1000)

    if random_bool() == 0:
        ans = result_plus(n1, n2)
        print(str(n1) + "+" + str(n2))
    else:
        ans = result_minus(n1, n2)
        print(str(n1) + "-" + str(n2))
    mistake = create_number_mistake(ans)
    print('ans_', ans)
    print('mistake_', mistake)
