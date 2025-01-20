# Required Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Create scatter plots for each pair of features
sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Create scatter plots for each pair of features
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for (i, (x, y)) in enumerate(pairs):
    ax = axes[i // 2, i % 2]
    sns.scatterplot(data=data, x=data.columns[x], y=data.columns[y], hue='species', ax=ax)
    ax.set_title(f'{data.columns[x]} vs {data.columns[y]}')

plt.tight_layout()
plt.show()
