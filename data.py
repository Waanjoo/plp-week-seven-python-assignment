import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
data = pd.read_csv('iris.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Group by species and calculate mean sepal length
grouped_data = data.groupby('species')['sepal_length'].mean()
print(grouped_data)

# Visualizations
# 1. Line plot (not applicable for Iris dataset)
# 2. Bar chart
plt.bar(grouped_data.index, grouped_data.values)
plt.xlabel('Species')
plt.ylabel('Mean Sepal Length')
plt.title('Mean Sepal Length by Species')
plt.show()

# 3. Histogram
plt.hist(data['sepal_length'], bins=20)
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Length')
plt.show()

# 4. Scatter plot
plt.scatter(data['sepal_length'], data['petal_length'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Scatter Plot of Sepal Length vs Petal Length')
plt.show()