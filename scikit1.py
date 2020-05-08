{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2051,)\n",
      "60000\n",
      "28\n",
      "28\n"
     ]
    }
   ],
   "source": [
    "import struct\n",
    "\n",
    "# ファイルを開く (rbはread binaryの意味)\n",
    "fp = open('train-images-idx3-ubyte', 'rb')\n",
    "\n",
    "# 4バイト読む\n",
    "b = fp.read(4)\n",
    "\n",
    "# バイトを整数に変換\n",
    "# MNISTの先頭4バイトはファイル識別用のマジックナンバー\n",
    "magic = struct.unpack('>i', b)\n",
    "print(magic)\n",
    "n_images, height, width = struct.unpack('>iii', fp.read(4 * 3))\n",
    "print(n_images)\n",
    "print(height)\n",
    "print(width)\n",
    "\n",
    "# ファイルを閉じる\n",
    "fp.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2051,)\n",
      "60000\n",
      "28\n",
      "28\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import struct\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt \n",
    "# ファイルを開く (rbはread binaryの意味)\n",
    "fp = open('train-images-idx3-ubyte', 'rb')\n",
    "\n",
    "# 4バイト読む\n",
    "b = fp.read(4)\n",
    "\n",
    "# バイトを整数に変換\n",
    "# MNISTの先頭4バイトはファイル識別用のマジックナンバー\n",
    "magic = struct.unpack('>i', b)\n",
    "print(magic)\n",
    "\n",
    "# マジックナンバー\n",
    "# 画像数、高さ、幅 (一気に複数の数字をunpackすることもできる)\n",
    "n_images, height, width = struct.unpack('>iii', fp.read(4 * 3))\n",
    "print(n_images)\n",
    "print(height)\n",
    "print(width)\n",
    "#pixelsのリスト\n",
    "pixels = struct.unpack('>' + 'B'*28*28, fp.read(1 * 28 * 28))\n",
    "pixels = np.asarray(pixels, dtype='uint8') #unit8:符号なし8ビット整数型\n",
    "pixels = pixels.reshape((height, width))\n",
    "\n",
    "plt.imshow(pixels, cmap='gray')\n",
    "plt.show()\n",
    "\n",
    "# ファイルを閉じる\n",
    "fp.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2051,)\n",
      "60000\n",
      "28\n",
      "28\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "\"\\nfor i in range(100):\\n    for j in range(100):\\n        index = i*100+j\\n        image = images[index]\\n        # 10x10のタイルのindex番目のaxisという意味\\n        # 番号は1から始まるので1を足しておく\\n        ax = fig.add_subplot(100, 100, index + 1)\\n        # グレースケールで表示\\n        ax.imshow(image, cmap='gray')\\n        # 画像の周りにある目盛りを消す\\n        ax.axis('off')\\nplt.show()\\n\""
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1440x1440 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import struct\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt \n",
    "# ファイルを開く (rbはread binaryの意味)\n",
    "fp = open('train-images-idx3-ubyte', 'rb')\n",
    "\n",
    "# 4バイト読む\n",
    "b = fp.read(4)\n",
    "\n",
    "# バイトを整数に変換\n",
    "# MNISTの先頭4バイトはファイル識別用のマジックナンバー\n",
    "magic = struct.unpack('>i', b)\n",
    "print(magic)\n",
    "\n",
    "# マジックナンバー\n",
    "# 画像数、高さ、幅 (一気に複数の数字をunpackすることもできる)\n",
    "n_images, height, width = struct.unpack('>iii', fp.read(4 * 3))\n",
    "print(n_images)\n",
    "print(height)\n",
    "print(width)\n",
    "\n",
    "images = []\n",
    "for i in range(10000):\n",
    "    pixels = struct.unpack('>' + 'B'*28*28, fp.read(1 * 28 * 28))\n",
    "    pixels = np.asarray(pixels, dtype='uint8')\n",
    "    pixels = pixels.reshape((height, width))\n",
    "    images.append(pixels)\n",
    "\n",
    "fig = plt.figure(figsize=(20,20))\n",
    "\"\"\"\n",
    "for i in range(100):\n",
    "    for j in range(100):\n",
    "        index = i*100+j\n",
    "        image = images[index]\n",
    "        # 10x10のタイルのindex番目のaxisという意味\n",
    "        # 番号は1から始まるので1を足しておく\n",
    "        ax = fig.add_subplot(100, 100, index + 1)\n",
    "        # グレースケールで表示\n",
    "        ax.imshow(image, cmap='gray')\n",
    "        # 画像の周りにある目盛りを消す\n",
    "        ax.axis('off')\n",
    "plt.show()\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2051,)\n",
      "60000\n",
      "28\n",
      "28\n",
      "10000\n",
      "100\n",
      "[[0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " ...\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]]\n",
      "0.92\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/shota/opt/miniconda3/envs/beginner/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)\n"
     ]
    }
   ],
   "source": [
    "fp = open('train-images-idx3-ubyte', 'rb')\n",
    "\n",
    "# 4バイト読む\n",
    "b = fp.read(4)\n",
    "\n",
    "# バイトを整数に変換\n",
    "# MNISTの先頭4バイトはファイル識別用のマジックナンバー\n",
    "magic = struct.unpack('>i', b)\n",
    "print(magic)\n",
    "\n",
    "# マジックナンバー\n",
    "# 画像数、高さ、幅 (一気に複数の数字をunpackすることもできる)\n",
    "n_images, height, width = struct.unpack('>iii', fp.read(4 * 3))\n",
    "print(n_images)\n",
    "print(height)\n",
    "print(width)\n",
    "n_images = len(images)\n",
    "print(n_images)\n",
    "images = np.asarray(images, dtype='uint8') #訳もわからず追加\n",
    "X = images.reshape((n_images, -1))\n",
    "X = X.astype('float32')\n",
    "#print(X)\n",
    "\n",
    "fp = open('train-labels-idx1-ubyte', 'rb')\n",
    "\n",
    "# バイトを整数に変換\n",
    "# MNISTの先頭4バイトはファイル識別用のマジックナンバー\n",
    "magic_l = struct.unpack('>i', fp.read(4))\n",
    "n_images_l = struct.unpack('>i', fp.read(4))\n",
    "y = struct.unpack('>'+'B'*10000, fp.read(10000))\n",
    "#print(y)\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "clf = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=10000).fit(X, y)\n",
    "\n",
    "#うまくいけばこれで学習が終わっている\n",
    "\n",
    "#ここからテスト画像\n",
    "fp = open('t10k-images-idx3-ubyte', 'rb')\n",
    "magic_t = struct.unpack('>i', fp.read(4))\n",
    "n_images_t, height_t, width_t = struct.unpack('>iii', fp.read(4 * 3))\n",
    "images_t = []\n",
    "for i in range(100):\n",
    "    pixels_t = struct.unpack('>' + 'B'*28*28, fp.read(1 * 28 * 28))\n",
    "    pixels_t = np.asarray(pixels_t, dtype='uint8')\n",
    "    pixels_t = pixels_t.reshape((height_t, width_t))\n",
    "    images_t.append(pixels_t)\n",
    "n_images_t = len(images_t)\n",
    "print(n_images_t)\n",
    "images_t = np.asarray(images_t, dtype='uint8') #訳もわからず追加\n",
    "X_test = images_t.reshape((n_images_t, -1))\n",
    "X_test = X_test.astype('float32')\n",
    "print(X_test)\n",
    "\n",
    "#ここからテストラベル\n",
    "fp = open('t10k-labels-idx1-ubyte', 'rb')\n",
    "\n",
    "# バイトを整数に変換\n",
    "# MNISTの先頭4バイトはファイル識別用のマジックナンバー\n",
    "magic_tl = struct.unpack('>i', fp.read(4))\n",
    "n_images_tl = struct.unpack('>i', fp.read(4))\n",
    "y_test = struct.unpack('>'+'B'*100, fp.read(100))\n",
    "#print(y_test)\n",
    "\n",
    "#ここまで\n",
    "\n",
    "#ここからテストデータに対する予測\n",
    "pred_labels = clf.predict(X_test)\n",
    "acc = (pred_labels == y_test).mean()\n",
    "\n",
    "\n",
    "print(acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0. 0. 0. 0. 0.]]\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'pytesseract' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-13-ad48a5ff5180>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mPIL\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mImage\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0mconfig\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m \u001b[0mtext\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpytesseract\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimage_to_string\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfromarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimage\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlang\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'eng'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mconfig\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mconfig\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;31m# 一文字のstringが得られる\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'pytesseract' is not defined"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "#raw_imgの読み込み\n",
    "for i in range(1,82):\n",
    "    filename = './raw_img/{0}.png'.format(i)\n",
    "    fp = open(filename,'rb')\n",
    "\n",
    "sudoku = np.zeros((9,9))\n",
    "print(sudoku)\n",
    "\n",
    "from PIL import Image\n",
    "config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'\n",
    "text = pytesseract.image_to_string(fp.fromarray(image), lang='eng', config=config)\n",
    "print(text) # 一文字のstringが得られる"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['0', '3', '5', '0', '9', '0', '0', '4', '8']\n",
      "['0', '0', '9', '0', '0', '8', '0', '0', '3']\n",
      "['0', '4', '0', '6', '0', '5', '0', '0', '1']\n",
      "['0', '0', '0', '0', '7', '4', '0', '0', '0']\n",
      "['0', '2', '0', '0', '0', '0', '0', '6', '0']\n",
      "['0', '0', '0', '1', '5', '0', '0', '0', '0']\n",
      "['8', '0', '0', '9', '0', '2', '0', '7', '0']\n",
      "['9', '0', '0', '5', '0', '0', '2', '0', '0']\n",
      "['6', '1', '0', '0', '4', '0', '5', '3', '0']\n"
     ]
    }
   ],
   "source": [
    "import copy\n",
    "from PIL import Image\n",
    "import sys\n",
    "import pyocr\n",
    "import pyocr.builders\n",
    "\n",
    "tools = pyocr.get_available_tools()\n",
    "if len(tools) == 0:\n",
    "    print(\"No OCR tool found\")\n",
    "    sys.exit(1)\n",
    "\n",
    "tool = tools[0]\n",
    "lang = 'eng'\n",
    "\n",
    "row_list = []\n",
    "res_list = []\n",
    "\n",
    "for x in range(1, 82):\n",
    "    text = tool.image_to_string(\n",
    "    Image.open(\"./raw_img/{}.png\".format(x)),\n",
    "    lang=lang,\n",
    "    # builder=pyocr.builders.DigitBuilder()\n",
    "    builder=pyocr.builders.DigitBuilder(tesseract_layout=6)\n",
    "    )\n",
    "    if text == \"\":\n",
    "        row_list.append(\"0\")\n",
    "    else:\n",
    "        row_list.append(text)\n",
    "    if x%9 == 0:\n",
    "        res_list.append(copy.deepcopy(row_list))\n",
    "        row_list = []\n",
    "\n",
    "for l in res_list:\n",
    "    print(l)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 3 5 0 9 0 0 4 8]\n",
      " [0 0 9 0 0 8 0 0 3]\n",
      " [0 4 0 6 0 5 0 0 1]\n",
      " [0 0 0 0 7 4 0 0 0]\n",
      " [0 2 0 0 0 0 0 6 0]\n",
      " [0 0 0 1 5 0 0 0 0]\n",
      " [8 0 0 9 0 2 0 7 0]\n",
      " [9 0 0 5 0 0 2 0 0]\n",
      " [6 1 0 0 4 0 5 3 0]]\n"
     ]
    }
   ],
   "source": [
    "problem = [[int(x) for x in y] for y in res_list]\n",
    "problem = np.array(problem)\n",
    "print(problem)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[2 3 5 7 9 1 6 4 8]\n",
      " [1 6 9 4 2 8 7 5 3]\n",
      " [7 4 8 6 3 5 9 2 1]\n",
      " [3 9 6 2 7 4 8 1 5]\n",
      " [5 2 1 3 8 9 4 6 7]\n",
      " [4 8 7 1 5 6 3 9 2]\n",
      " [8 5 3 9 6 2 1 7 4]\n",
      " [9 7 4 5 1 3 2 8 6]\n",
      " [6 1 2 8 4 7 5 3 9]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "import solver\n",
    "solver.sudoku_solve(problem)\n",
    "print(problem)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
