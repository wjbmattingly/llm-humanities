# Leveraging Large Language Models for Humanities Data

Over the past few years, large language models (LLMs), like ChatGPT and Claude, have become instrumental in performing various natural language processing (NLP) tasks. They are proficient in natural language understanding and can produce quality responses to various NLP problems. Despite this, LLMs need to be guided and prompted in specific ways to have optimized and consistent outputs. In this course, we will learn how to leverage LLMs to classify named entities in texts through zero-shot, single-shot, and few-shot classification. We will learn how to ensure consistent structure in our outputs via OpenAI's new line of GPT-4o models and Pydantic. Most importantly, we will learn how to leverage these outputs to frame and answer humanities-specific qualitative and quantitative questions.

## Schedule

### Day 1: Introduction to Key Concepts & Setting up OpenAI
- Understanding Large Language Models (LLMs) and their capabilities
  - Overview of ChatGPT, Claude, and GPT-4
  - Key differences between model generations
  - Strengths and limitations in humanities research
- Setting up OpenAI API access
  - Creating an API key
  - Basic API calls and rate limits
  - Cost considerations and best practices
- Introduction to prompt engineering
  - Zero-shot, one-shot, and few-shot learning
  - Crafting effective prompts
  - Handling context and token limitations

### Day 2: Importance of Structured Data
- Working with the American Stories dataset
  - Dataset overview and access via Hugging Face
  - Data quality and OCR considerations
- Structured outputs with Pydantic
  - Creating data models
  - Validation and error handling
  - Converting LLM responses to structured formats
- Named Entity Recognition (NER) tasks
  - Identifying people, places, and organizations
  - Temporal expressions and dates
  - Domain-specific entity types

### Day 3: Solving a Domain-Specific Problem
- Case studies in humanities research
  - Topic classification of historical articles
  - Content deduplication across newspapers
  - Analyzing narrative patterns and themes

## Dataset

The main dataset that we will be working with in this course is [American Stories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories). This dataset contains high-quality digitized article texts extracted from nearly 20 million historical U.S. newspaper scans from the Library of Congress's Chronicling America collection. The dataset was created using a novel deep learning pipeline that includes layout detection, legibility classification, custom OCR, and article text association.

Key features of the dataset:
- Contains 1.14 billion content regions including articles, headlines, captions, and advertisements
- Extensive geographic coverage across all U.S. states
- Content dating from the 17th century through early 20th century
- High-quality OCR with character error rates under 5%
- Structured article-level texts enabling advanced NLP applications
- Creative Commons CC-BY license

The dataset can be easily accessed using the Hugging Face datasets library:

```python
from datasets import load_dataset

# Load data for specific years at article level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "subset_years",
    year_list=["1939", "1940", "1941", "1942", "1943", "1944", "1945"]
)

# Load all years at article level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years"
)
```

The dataset provides an invaluable resource for studying historical texts, training language models, and developing multimodal applications. The structured article texts enable advanced NLP tasks like topic classification, content deduplication, and news story clustering.

For more details about the dataset and its creation, see the [original paper](https://arxiv.org/abs/2308.12477). 