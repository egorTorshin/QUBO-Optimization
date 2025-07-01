def check_minimum(array1: list, array2: list) -> float:
    mn = 10 ** 10

    for i in range(len(array1)):
        mn = min(mn, array2[i] - array1[i])

    return mn
