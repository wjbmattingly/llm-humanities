from datasets import load_dataset, Dataset
import random
from huggingface_hub import HfApi, HfFolder
import os

dataset = load_dataset("wjbmattingly/american-stories-sample", split="train")



print(dataset[0])

# Filter articles that mention specific names and print their lengths
names_to_search = ["Hitler", "Stalin", "Roosevelt", "Churchill"]

# Initialize a list to store filtered articles
filtered_articles = []

# Iterate over the dataset
for article in dataset:
    # Check if any of the names are in the article text
    if any(name in article['article'] for name in names_to_search):
        # Append the article to the filtered list
        filtered_articles.append(article)

# Create a new dataset from the filtered articles
new_dataset = Dataset.from_dict({'article': [article['article'] for article in filtered_articles]})

# Print the number of articles in the new dataset
print(f"Number of filtered articles: {len(new_dataset)}")

# Set random seed for reproducibility
random.seed(42)

# Initialize lists to store articles for each person
hitler_articles = []
stalin_articles = []
roosevelt_articles = []
churchill_articles = []

# Iterate over the dataset
for article in dataset:
    # Check if any of the names are in the article text and append to respective list
    if "Hitler" in article['article']:
        hitler_articles.append(article)
    elif "Stalin" in article['article']:
        stalin_articles.append(article)
    elif "Roosevelt" in article['article']:
        roosevelt_articles.append(article)
    elif "Churchill" in article['article']:
        churchill_articles.append(article)

# Sample 250 articles from each list (or as many as available)
sampled_articles = (
    random.sample(hitler_articles, min(250, len(hitler_articles))) +
    random.sample(stalin_articles, min(250, len(stalin_articles))) +
    random.sample(roosevelt_articles, min(250, len(roosevelt_articles))) +
    random.sample(churchill_articles, min(250, len(churchill_articles)))
)

# Create a new dataset from the sampled articles, including all fields
sampled_dataset = Dataset.from_dict({
    key: [article[key] for article in sampled_articles] for key in sampled_articles[0].keys()
})

# Save the dataset as a Parquet file
sampled_dataset.to_parquet('sample_articles.parquet')

# Print the number of articles in the sampled dataset
print(f"Number of sampled articles: {len(sampled_dataset)}")

# Initialize Hugging Face API
api = HfApi()

# Define repository name and type
repo_name = "wjbmattingly/american-stories-sample-tap"
repo_type = "dataset"


try:
    api.repo_info(repo_name, repo_type=repo_type)
    print(f"Repository {repo_name} already exists.")
except Exception as e:
    # Create the repository if it doesn't exist
    api.create_repo(repo_name, repo_type=repo_type, exist_ok=True)
    print(f"Created repository {repo_name}.")

# Upload the Parquet file to the repository
parquet_file_path = 'sample_articles.parquet'
api.upload_file(
    path_or_fileobj=parquet_file_path,
    path_in_repo="sample_articles.parquet",
    repo_id=repo_name,
    repo_type=repo_type
)

print(f"Uploaded {parquet_file_path} to {repo_name}.")

