# 1 - After preprocessing & formulated into sparse matrix - creating nmf
import pandas as pd
from scipy.sparse import csr_matrix

df = pd.read_csv('wiki.csv', index_col=0) # looks like it has top 60
articles = csr_matrix(df.transpose())
titles = list(df.columns)

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
# print(nmf_features)

# 2 -
# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features,index=titles)

print(df)
# Print the row for 'Anne Hathaway'
# print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
# print(df.loc['Denzel Washington'])

# Import pandas
import pandas as pd

words = pd.read_csv('wiki_titles.txt', header=None) # words file
words = words.iloc[:,0].values

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_,columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3,:]

# Print result of nlargest
print(component.nlargest()) #gives highest values for that component.

