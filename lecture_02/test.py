import pandas as pd
import matplotlib.pyplot as plt

# Read file
df = pd.read_csv('train.csv')

# Display the summary statistics
print(df.describe())


# Count the number of rows with at least one empty value
rows_with_empty_values = df[df.isnull().any(axis=1)]

# Print the number
print("There are " + str(len(rows_with_empty_values)) + " rows with at least one empty value.")


# remove columns with more than 200 NaN values
df = df.dropna(axis=1, thresh=200)

# Display the remaining columns
print(df.columns)


# replace 'male' with 0 and 'female' with 1 in the 'Sex' column
df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})

# Display the first few rows of the DataFrame
print(df.head())


# Extract and split 'Name' into 'First Name', 'Middle Name', 'Last Name', and 'Title'
name_split = df['Name'].str.extract(r'(?P<Last_Name>[\w\s]+),\s(?P<Title>\w+)\.\s(?P<First_Name>[\w\s]+)\s?(?P<Middle_Name>[\w\s]+)?')

# Assign the extracted values to the new columns
df[['First Name', 'Middle Name', 'Last Name', 'Title']] = name_split[['First_Name', 'Middle_Name', 'Last_Name', 'Title']]

print(df.head())


# Replace missing values in 'Age' with the average age
average_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(average_age)

print(df.head())


# Group by 'Survived' and calculate the average age for each group
average_age_by_survived = df.groupby('Survived')['Age'].mean()

# Plotting the bar chart
average_age_by_survived.plot(kind='bar', color=['red', 'green'])
plt.title('Average Age by Survival Status')
plt.xlabel('Survived')
plt.ylabel('Average Age')
plt.xticks([0, 1], ['Did not Survive', 'Survived'], rotation=0)
plt.show()