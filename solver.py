import struct
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from IPython.display import Image, display


def sudoku_solve(problem):
    rows, cols = problem.shape
    all_nums = set([i for i in range(0, 10)])
    for i in range(rows):
        for j in range(cols):
            if problem[i, j] == 0:
                # 候補の数字を調べる (以下のnumsに代入する)
                k = 3 * (i // 3)
                l = 3 * (j // 3)
                blk_nums = problem[k:k+3, l:l+3]
                row_nums = problem[i,:]
                col_nums = problem[:,j]
                used_nums = set(row_nums.tolist() + col_nums.tolist() + blk_nums.flatten().tolist())
                nums = set([i for i in range(10)]).difference(used_nums)

                # 候補の数字を仮入れして再帰呼び出し
                success = False
                for n in nums:
                    # 数字を代入
                    problem[i, j] = n
                    if sudoku_solve(problem):
                        success = True
                        break

                    # 失敗したら元に戻す
                    problem[i, j] = 0

                if not success:
                    return False
        
    return True