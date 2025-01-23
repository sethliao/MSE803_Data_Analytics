################ Data cleaning the Iris dataset #################
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe
iris_df = pd.DataFrame(iris.data)
iris_df["class"] = iris.target

# Add column names to the dataframe
iris_df.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]

#### ===> TASK 1: here - add two more lines of the code to find the number and mean of missing data
missing_values = iris_df.isnull().sum()
mean_values = iris_df.mean()

# Print missing values and mean values
print("Missing values in each column:\n", missing_values)
print("\nMean of each column:\n", mean_values)

# remove any empty lines
iris_df.dropna(how="all", inplace=True)


# Select the first 5 rows and the first 4 columns of the dataset
iris_X = iris_df.iloc[:5, [0, 1, 2, 3]]
print("\nFirst 5 rows of feature values:\n")
print(iris_X)

### TASK2: Here - Write a short readme to explain above code and how we can calculate the corrolation amoung featuers with description

#### Feature correlation matrix
correlation_matrix = iris_df.corr()
print("\nFeature correlation matrix (Pearson correlation):\n", correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix (Iris Dataset)")
plt.show()
