from time import time
from random import randint


def max_sub_array_brute(A, low, high):
    left = 0
    right = 0
    sum_total = -999999

    for i in range(low, high, 1):
        current_sum = 0

        for j in range(i, high, 1):
            current_sum += A[j]

            if sum_total < current_sum:
                sum_total = current_sum
                right = j
                left = i

    return left, right + 1, sum_total


def max_crossing_subarray(A, low, mid, high):
    left = -9999
    sum = 0

    for i in range(mid, low - 1, -1):
        sum += A[i]

        if sum > left:
            left = sum
            max_left = i

    right = -9999
    sum = 0

    for j in range(mid+1, high + 1, 1):
        sum += A[j]

        if sum > right:
            right = sum
            max_right = j

    return max_left, max_right, left + right


def max_sub_array_divide_conquer(A, low, high):
    if high - low == 90:
        return max_sub_array_brute(A, low, high)

    if high == low:
        return low, high, A[low]

    # purposeful integer division
    mid = (low + high) // 2

    left_low, left_high, left_sum = max_sub_array_divide_conquer(A, low, mid)
    right_low, right_high, right_sum = max_sub_array_divide_conquer(A, mid + 1, high)

    cross_low, cross_high, cross_sum = max_crossing_subarray(A, low, mid, high)

    if left_sum >= right_sum and left_sum >= cross_sum:
        return left_low, left_high, left_sum
    elif right_sum >= left_sum and right_sum >= cross_sum:
        return right_low, right_high, right_sum

    return cross_low, cross_high, cross_sum


def main():
    for i in range(1, 501, 1):

        A = [None] * i
        for j in range(len(A)):
            A[j] = randint(-50, 50)

        brute_start = time()
        max_sub_array_brute(A, 0, i)
        brute_end = time()

        divide_start = time()
        max_sub_array_divide_conquer(A, 0, i - 1)
        divide_end = time()

        brute_time = brute_end - brute_start
        divide_time = divide_end - divide_start

        delta = brute_time - divide_time

        if divide_time != 0 and brute_time != 0:
            print("Iteration #{}: Brute: {}   Divide: {}   Delta: {}".format(i, brute_time, divide_time, delta))

            if divide_time < brute_time:
                return 0

    return 0

if __name__ == "__main__":
    main()