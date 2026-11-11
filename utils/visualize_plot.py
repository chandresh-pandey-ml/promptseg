import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('logfiles/joint_lora_training_log.csv')

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Average Loss', color=color, fontsize=12)
ax1.plot(df['Epoch'], df['Average_Loss'], color=color, marker='o', linewidth=2, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Learning Rate', color=color, fontsize=12)
ax2.plot(df['Epoch'], df['Learning_Rate'], color=color, linestyle='--', linewidth=2, label='Learning Rate (Cosine)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Convergence and Learning Rate Schedule', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.show()