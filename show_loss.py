
from tbparse import SummaryReader
import plotext as plt

log_file = "/home/uqhyan14/vlm_challenge/CDSegNet/exp/nuscenes/CDSegNet/events.out.tfevents.1756637704.bun068"

# pivot=True 默认就是这种 DataFrame 格式
reader = SummaryReader(log_file, pivot=True)
df = reader.scalars

print("可用指标列:", df.columns.tolist())
# 去掉包含 NaN 的行
df = df.dropna(subset=["train/loss", "val/loss"])
steps = df["step"].tolist()
train_loss = df["train/loss"].tolist()
val_loss = df["val/loss"].tolist()
val_iou = df["val/mIoU"].tolist()
print("train_loss:",train_loss)
print("val_loss:",val_loss)
print("val_iou:",val_iou)

plt.clt()  # 清空旧图
plt.plotsize(100, 30)  # 画布大小

plt.plot(steps, train_loss, label="Tl")
plt.plot(steps, val_loss, label="Vl")
# plt.plot(steps, val_iou, label="Vu")


plt.title("Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")

# 设置横坐标只显示已有的 steps
plt.xticks(steps)

plt.show()