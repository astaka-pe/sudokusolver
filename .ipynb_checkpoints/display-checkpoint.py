import copy
from PIL import Image
import sys
import pyocr
import pyocr.builders
import numpy as np

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

tool = tools[0]
lang = 'eng'

row_list = []
res_list = []

for x in range(1, 82):
    text = tool.image_to_string(
    Image.open("./raw_img2/{}.png".format(x)),
    lang=lang,
    # builder=pyocr.builders.DigitBuilder()
    builder=pyocr.builders.DigitBuilder(tesseract_layout=6)
    )
    if text == "":
        row_list.append("0")
    else:
        row_list.append(text)
    if x%9 == 0:
        res_list.append(copy.deepcopy(row_list))
        row_list = []
"""
for l in res_list:
    print(l)
"""
problem = [[int(x) for x in y] for y in res_list]
problem = np.array(problem)
print(problem)

import solver
solver.sudoku_solve(problem)
print(problem)