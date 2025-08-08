import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from SalUn JSON
with open("models/inceptionv3_cat_forgetting/salun_eval_results.json") as f:
    results = json.load(f)

# Extract values
labels = ["Forget", "Retain"]
train_accuracies = [
    results["train_forget_accuracy"],
    results["train_retain_accuracy"]
]
test_accuracies = [
    results["test_forget_accuracy"],
    results["test_retain_accuracy"]
]

# Plotting
x = np.arange(len(labels))  # [0, 1]
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, train_accuracies, width, label='Train', color='skyblue')
rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test', color='lightcoral')

# Labels, title, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Forget vs. Retain Accuracy (SalUn)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.legend()

# Add text on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4pt vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("salun_forget_retain_accuracy.png")
plt.show()
