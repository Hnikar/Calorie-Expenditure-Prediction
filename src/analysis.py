import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_data():
    train_df = pd.read_csv('data/train.csv')
    
    os.makedirs('reports', exist_ok=True)
    
    desc = train_df.describe()
    with open('reports/data_analysis.txt', 'w') as f:
        f.write("Basic Statistics:\n")
        f.write(desc.to_string())
        f.write("\n\nMissing Values:\n")
        f.write(train_df.isnull().sum().to_string())

    plt.figure(figsize=(10, 8))
    numeric_df = train_df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('reports/correlation_matrix.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(train_df['Calories'], kde=True)
    plt.title('Distribution of Calories')
    plt.savefig('reports/calories_dist.png')
    plt.close()
    
    print("Analysis complete. Reports saved to the reports folder/")

if __name__ == "__main__":
    analyze_data()