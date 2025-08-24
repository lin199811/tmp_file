import pickle

infile = "/mnt/e/Lin/Downloads/Compressed/track5/source/resnet_lion/result.pkl"
outfile = "/mnt/e/Lin/Downloads/Compressed/track5/source/resnet_lion/result_shift.pkl"

with open(infile, "rb") as f:
    data = pickle.load(f)

# 遍历每一帧，把 pred_labels +1
for frame in data:
    frame["pred_labels"] = frame["pred_labels"] + 1

# 保存到新的文件（避免覆盖原始文件）
with open(outfile, "wb") as f:
    pickle.dump(data, f)

print(f"修改完成，保存到 {outfile}")
