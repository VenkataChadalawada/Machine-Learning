import pandas as pd
samples = pd.read_csv('stocks.csv')
samples.head()

companies = samples.iloc[:,0].values
movements = samples.iloc[:,1:].values

# Import Normalizer, make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()
# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

