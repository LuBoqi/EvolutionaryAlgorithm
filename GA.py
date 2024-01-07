import numpy as np
import random

# 默认找最大值，如需更改为寻找最小值，将evaluation加负号

varNum = 2  # 变量数目
rank = 32  # 编码位数
low = np.array([-5, -5])  # 变量下界
high = np.array([5, 5])  # 变量上界
epochs = 100  # 迭代次数
population = 20  # 群体数目
crossoverRate = 0.8  # 交叉率
mutateRate = 0.1  # 变异率


def evaluation(codes):
    x = decode(codes)
    # f = -x[0] ** 2 + 10 * np.cos(2 * np.pi * x[0]) + 30
    # f = x[0] ** 2
    f = (20 + np.e - 20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) -
         np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))))
    return f


def encode(nums):
    precision = (high - low) / (np.power(2, rank) - 1)
    for i in range(len(nums)):
        nums[i] = (nums[i] - low[i]) / precision[i]
    code = np.vectorize(np.binary_repr)(nums.astype(int), width=rank)
    return code


def decode(codes):
    precision = (high - low) / (np.power(2, rank) - 1)
    nums = codes.copy().astype(float)
    for i in range(len(codes)):
        for j in range(len(codes[0])):
            nums[i][j] = int(codes[i][j], 2) * precision[i] + low[i]
    return nums


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 避免指数溢出，减去最大值
    return exp_x / exp_x.sum(axis=0, keepdims=True)


def selection(codes, quantity):
    vals = evaluation(codes)
    index = np.argsort(vals)[::-1]
    vals = softmax(vals)
    selects = [[] for _ in range(len(codes))]
    for i in range(len(codes)):
        selects[i].append(codes[i][index[0]])
    for k in range(len(codes)):
        for i in range(quantity - 1):
            aRate = random.random()
            cumulative = 0
            for j in range(len(vals)):
                cumulative += vals[j]
                if aRate < cumulative:
                    selects[k].append(codes[k][j])
                    break
    return np.array(selects)


def crossover(codes):
    re_codes = codes.copy()
    for k in range(len(codes)):
        for i in range(len(codes[0])):
            if random.random() < crossoverRate:
                select_idx = random.randint(0, len(codes) - 1)
                while select_idx == i:
                    select_idx = random.randint(0, len(codes[0]) - 1)
                select_code = codes[k][select_idx]
                crossPos = random.randint(1, len(select_code) - 2)
                re_codes[k][i] = codes[k][i][:crossPos] + select_code[crossPos:]
    return re_codes


def mutation(codes):
    re_codes = codes.copy()
    for k in range(len(codes)):
        for i in range(len(codes[0])):
            code = list(str(re_codes[k][i]))
            for j in range(len(code)):
                if random.random() < mutateRate:
                    if code[j] == '1':
                        code[j] = '0'
                    else:
                        code[j] = '1'
            re_codes[k][i] = ''.join(code)
    return re_codes


def geneticAlgorithms():
    nums = np.zeros((varNum, population))
    for i in range(varNum):
        nums[i, :] = np.random.uniform(low[i], high[i], population)
    codes = encode(nums)
    for epoch in range(epochs):
        origCode = codes.copy()
        codes = crossover(codes)
        codes = mutation(codes)
        codes = np.hstack((origCode, codes))
        codes = np.unique(codes, axis=1)
        codes = selection(codes, population)
    vals = evaluation(codes)
    index = np.argsort(vals)[::-1]
    ans = [[] for _ in range(len(codes))]
    for i in range(len(codes)):
        ans[i].append(codes[i][index[:3]])
    ans = np.reshape(np.array(ans), (varNum, -1))
    return evaluation(ans), decode(ans), ans


if __name__ == '__main__':
    results = geneticAlgorithms()
    for result in results:
        print(result)
        print('------------------------------------------------')
