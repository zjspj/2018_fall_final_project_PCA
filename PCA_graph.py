import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
features = ['0M', '1M', '3M', '6M', '12M']
x = df.loc[:, features].values
y = df.loc[:,['Amino acids']].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
ratios = []
for ratio in pca.explained_variance_ratio_:
    ratio_str = str(ratio*100)
    ratios.append(ratio_str[:4])
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", 'PC2'])
finalDf = pd.concat([principalDf, df[['Amino acids']]], axis =1)
finalDf.drop(finalDf.tail(1).index,inplace = True)
fig = plt.subplots(figsize=(8,8))
pca_plot = sns.scatterplot(x='PC1',
                           y='PC2',
                           hue='Amino acids',
                           data=finalDf,
                           sizes = (20,20),
                           )
plt.ylim(-1.2,1.2)
plt.xlim(-1.2,1.2)
plt.setp(pca_plot.get_legend().get_texts(), fontsize="7")
plt.setp(pca_plot.get_legend().get_title(), fontsize="10")
pca_plot.axhline(0,ls='-',color="k")
pca_plot.axvline(0,ls='-',color="k")
pca_plot.set_xlabel("PC1" + " (" + ratios[0] + "%)", fontsize = 15)
pca_plot.set_ylabel("PC2" + " (" + ratios[1] + "%)", fontsize = 15)
pca_plot.set_title('PCA', fontsize = 20)
plt.show()
