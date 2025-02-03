from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# read the cv file
data = pd.read_csv('tracks.csv')

data.shape

data.info()

data.describe().transpose()

# Select numeric columns only
numeric_data = data.select_dtypes(include=['number'])

# Generate the summary for numeric columns
numeric_summary = numeric_data.describe().transpose()
pd.set_option('display.max_columns', None)

# Display the summary
print("\nSummery for numeric columns:")
print(numeric_summary)

#--------------------------data cleaninng---------------------------------#
print("\nShow the number of missing values in each column:")
print(data.isnull().sum())

print("\nCheck if DataFrame has duplicated rows:")
print(data.duplicated().sum())
# fill the empty slots
data['name'] = data['name'].fillna('Unknown')



#-----clean 'artists' name-------------------#
print("\nOriginal 'artists' column (first 10 rows):")
print(data['artists'][0:10])  # View the first 10 rows before cleaning

# Extract the actual artist names and remove the square brackets and quotes
data['artists'] = data['artists'].str.strip("[]").str.replace("'", "")
print("\nCleaned 'artists' column (first 10 rows):")
print(data['artists'][0:10])  # View the cleaned artist names

#-----convert the units of 'duration'-----#
print("\nOriginal 'duration_ms' column (first 10 rows):")
print(data['duration_ms'][0:10])  # View the first 10 rows before conversion

# Convert duration from milliseconds to seconds
data['duration_ms'] = data['duration_ms'].apply(lambda x: round(x/1000))

# Rename duration column
data.rename(columns={'duration_ms': 'duration'}, inplace=True)
print("\nConverted and renamed 'duration' column (first 10 rows):")
print(data['duration'][0:10])  # View the converted and renamed column

#------convert datatype of 'release date'--#
print("\nOriginal 'release_date' column (first 10 rows):")
print(data['release_date'][0:10])  # View the first 10 rows before conversion

# Change the data type to datetime format
data['release_date'] = pd.to_datetime(data['release_date'], format='mixed')
print("\nConverted 'release_date' column (first 10 rows):")
print(data['release_date'][:10])  # View the converted 'release_date'


# save the clean version
data.to_csv('cleaned_dataset_tracks.csv', index=False)
print("Data cleaned and saved successfully!")

cleaned_data = pd.read_csv('cleaned_dataset_tracks.csv')

#------------------------------------------------------------------------#

#-------------------------Normalize data---------------------------------#


# Load the cleaned dataset
data = pd.read_csv('cleaned_dataset_tracks.csv')
# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply the scaler to the numeric data
normalized_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)

# Display the normalized data
print("\nHere is the normalized data:")
print(normalized_data.head())

# Optionally, save normalized data to a new CSV file
normalized_data.to_csv('normalized_numeric_data.csv', index=False)

#-------------------------------------------------------------------------#

#------------------------Correlation Analysis-----------------------------#

# Select only numerical columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix for numerical data
correlation_matrix = numeric_data.corr()

# Sort correlations with 'popularity' in descending order
popularity_correlation = correlation_matrix['popularity'].sort_values(ascending=False)

# Display the correlations
print("\nCorrelations with Popularity:")
print(popularity_correlation)

# Plot the heatmap for correlations with popularity
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f",cmap='coolwarm')
plt.title("Correlation with Popularity")
plt.show()

print("\nCorrelation Analysis Interpretation:")
print("The correlation analysis reveals which song features influence popularity the most.")
print("If 'energy' or 'danceability' have high positive correlations, they likely contribute to popular songs.")
print("A strong negative correlation might indicate that a certain attribute reduces popularity.")
print(f"Top correlated attribute: {popularity_correlation.index[1]} with {popularity_correlation.iloc[1]:.2f} correlation.")

# Bar plot for the top correlations
plt.figure(figsize=(10, 6))
popularity_correlation.drop('popularity').plot(kind='bar', color='darkblue')
plt.title("Track Attributes Correlation with Popularity")
plt.ylabel("Correlation Coefficient")
plt.xlabel("Attributes")
plt.show()



# Convert release_date to datetime and extract year
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year

# Group by release year to analyze track release patterns
tracks_per_year = data.groupby('release_year').size()

# Plot number of tracks released over time
plt.figure(figsize=(10, 6))
tracks_per_year.plot(kind='line', title='Tracks Released Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Tracks')
plt.show()

# Analyze engagement metrics (average popularity)
popularity_over_time = data.groupby('release_year')['popularity'].mean()

# Plot average popularity over time
plt.figure(figsize=(10, 6))
popularity_over_time.plot(kind='line', title='Average Popularity Over Time')
plt.xlabel('Year')
plt.ylabel('Average Popularity')
plt.show()

#----------------------------------------------------------------------------#

# Load the normalized dataset
normalized_data = pd.read_csv('normalized_numeric_data.csv')

# Add the original dataset for non-numeric columns if needed
original_data = pd.read_csv('cleaned_dataset_tracks.csv')
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data)

explained_variance_ratio = pca.explained_variance_ratio_

# PCA Scatter Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=normalized_data['popularity'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Popularity')
plt.title('PCA Scatter Plot of Songs')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title="Popularity")  
plt.show()

print("\nPCA Analysis Interpretation:")
print("The PCA scatter plot reduces the dataset to two principal components.")
print("If clusters appear, it indicates that songs with similar features group together.")
print(f"PCA Explained Variance: {explained_variance_ratio[0]:.2f} and {explained_variance_ratio[1]:.2f}.")
print("This means that these two components capture most of the variance in the data.")

# t-SNE: Perform dimensionality reduction
sampled_data = normalized_data.sample(10000, random_state=42).copy()
tsne = TSNE(n_components=2, perplexity=20, max_iter=500, random_state=42)
tsne_result = tsne.fit_transform(sampled_data)

# t-SNE Scatter Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=sampled_data['popularity'], cmap='plasma', alpha=0.5)
plt.colorbar(label='Popularity')
plt.title('t-SNE Scatter Plot of Songs')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(*scatter.legend_elements(), title="Popularity")  
plt.show()

print("\n t-SNE Analysis Interpretation:")
print("The t-SNE scatter plot visualizes the dataset in a non-linear fashion, emphasizing local relationships.")
print("Clusters in this plot indicate groups of songs that share similar features, such as tempo, energy, or danceability.")
print("If clear clusters appear, it suggests that the dataset has strong patterns and song similarities.")
print("If there is a smooth transition of colors, it may indicate a continuous trend in song popularity.")
print("Isolated points suggest unique or outlier songs with distinct characteristics.")


# PCA KMeans Clustering
inertia_pca = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_result)
    inertia_pca.append(kmeans.inertia_)

# PCA Elbow Plot
plt.figure(figsize=(10, 6))
plt.plot(K, inertia_pca, 'bo-', linewidth=2)
plt.title('Elbow Method for Optimal k (PCA Reduced Data)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Apply KMeans Clustering on PCA
k_pca = 4  # Choose optimal k based on the elbow plot
kmeans_pca = KMeans(n_clusters=k_pca, random_state=42)
clusters_pca = kmeans_pca.fit_predict(pca_result)

# Add cluster labels to the original data
normalized_data['Cluster_PCA'] = clusters_pca

# Visualize PCA Clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters_pca, cmap='tab10', alpha=0.7)
plt.colorbar(label='Cluster')
plt.title('KMeans Clustering Visualization (PCA Reduced Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title="Clusters") 
plt.show()

print("\nK-Means Clustering (PCA) Interpretation:")
print("The K-Means clustering groups songs based on their musical features.")
print(f"Number of clusters: {k_pca}.")
print("If one cluster has mostly high-popularity songs, it means those songs share similar traits.")
print("If the clusters are well-separated, it suggests distinct song categories.")

# PCA Cluster Summary
numeric_columns = normalized_data.select_dtypes(include=['number']).columns
cluster_summary_pca = normalized_data.groupby('Cluster_PCA')[numeric_columns].mean()
print("\nPCA Cluster Summary:")
print(cluster_summary_pca)

# t-SNE KMeans Clustering
inertia_tsne = []
K = range(1, 11)

# Apply KMeans Clustering on t-SNE
k_tsne = 4  # Choose optimal k based on the elbow plot
kmeans_tsne = KMeans(n_clusters=k_tsne, random_state=42)
clusters_tsne = kmeans_tsne.fit_predict(tsne_result)

# Add cluster labels to the sampled data
sampled_data['Cluster_TSNE'] = clusters_tsne

# Visualize t-SNE Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    hue=clusters_tsne,
    palette='tab10',
    alpha=0.7
)
plt.title('KMeans Clustering Visualization (t-SNE Reduced Data)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster')  
plt.grid(True)
plt.show()


print("\nK-Means Clustering (t-SNE) Interpretation:")
print("The t-SNE visualization shows the clusters in a non-linear way, capturing complex relationships.")
print(f"Number of clusters: {k_tsne}.")
print("If clusters are overlapping, it means songs share mixed characteristics.")
print("If distinct clusters form, they represent different song styles or genres.")

# t-SNE Cluster Summary
numeric_columns_sampled = sampled_data.select_dtypes(include=['number']).columns
cluster_summary_tsne = sampled_data.groupby('Cluster_TSNE')[numeric_columns_sampled].mean()
print("\nt-SNE Cluster Summary:")
print(cluster_summary_tsne)