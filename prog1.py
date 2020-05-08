import struct

# ファイルを開く (rbはread binaryの意味)
fp = open('binary_file.bin', 'rb')

# 4バイト読む
b = fp.read(4)

# バイトを整数に変換
# MNISTの先頭4バイトはファイル識別用のマジックナンバー
magic = struct.unpack('>i', b)
print(magic)

# ファイルを閉じる
fp.close()