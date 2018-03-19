# Load .mat file
from scipy.io import loadmat

## File is originally a dictionary
data_dict = loadmat('one_to_ten.mat')
print('Data type', type(data))
print('Data:\n', data_dict, '\n---------------')
print('Datakeys:\n', data_dict.keys() )

## We have to index by the variable name
var_name = 'one2ten'
data = data_dict[var_name]
print(data)

## Now to a more interesting (or not) dataset
data = loadmat('iris_dataset.py')
X, y = data['X'], data['y']

print('X: ', type(X), X.shape)
print('y: ', type(y), y.shape)

# Visualize distribution
import matplotlib.pyplot as plt
import seaborn as sns

## We could visualize


## Using seaborn is easier
sns.pairplot(vars=["one","two","three"], data=df, hue="label", size=5)
plt.show()

# Still, we want to visualize the distribution, in only two dimensions

## Lets PCA
from sklearn.decomposition import PCA

### PCA is not a function, is a class, but you will get used to it
### This means it has to be initialized by itself, in despite of the dataset
pca = PCA()
### The dataset has to be organized with one observation per row
### and one feature per columns, i.e. (n_observations, n_features)
pca.fit(X)
### The fit method is used by all sklearn models
### Some of then have also the 'predict' method, and some have 'tr==ansform'
Xpc = pca.transform(X)
print('Returned dataset format: ', Xpc.shape)

### Is the dataset still the same? lets pairplot
df =
sns.pairplot(vars=["one","two","three"], data=df, hue="label", size=5)
plt.show()

### The returned features are the data projected into the principal components,
### and not the original features
print(X.sum(axis=0) == Xpc.sum(axis==0))

# Lets then predict the classes from the data
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


## Choose a model
clf =

## Fit the model (1 line) (now we will do the wrong way - no splitting)


## Predict


## Look at results
