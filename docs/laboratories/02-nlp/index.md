---
authors: balazskvancz
---

# 02 - NLP and Embedding-based Applications

## Goal

The main goal of this laboratory is to become familiar with the basic tools used for Natural Language Processing (NLP) tasks.

The structure of this laboratory is as follows:

1. Simple Semantic Similarity Search Engine [3 points in total]
    1. Computing sentence embeddings. [1 point]
    1. Determining which sentences in the dataset are most semantically similar to the query. [1 point]
    1. Filter the `top_k` elements. [1 point]
1. Clustering and Visualisation of Sentences [4 points in total]
    1. Computing sentence embeddings. [1 point]
    1. Grouping the sentences into `k` clusters based on their semantic similarity. [1 point]
    1. Applying dimension reduction into two-dimensional space. [1 point]
    1. Visualising the clustered data. [1 point]
1. Multi-class News Categorization [6 points in total]
    1. Loading the dataset properly. [1 point]
    1. Implementing vector embedding computation function. [2 points]
    1. Classifying the vector embeddings using `Logistic Regression` models. [2 points]
    1. Evaluating the results. [1 point]
    1. Visualising embeddings in 2D – UMAP – with points colored by predicted class. [2 points]
1. Paraphrase Detection using Sentence Embeddings [6 points in total]
    1. Loading the dataset properly. [1 point]
    1. Implementing vector embedding computation function. [2 points]
    1. Calculating cosine similarity and turning them into predictions. [1 point]
    1. Evaluating the results. [1 point]
    1. Plotting the distribution of similarity scores. [1 point]
1. Giving feedback [+1 point]

!!! info "Grading"
    In order to pass this laboratory, you must obtain at least 8 points out of 19.

!!! important "Screenshot requirement"
    At the end of **every** exercise (marked with a :camera: **Screenshot** note), you must take a screenshot of your working solution, including its output, and upload it together with your solution, using the exact filename given in that note (e.g. `f1.png`). Solutions missing the required screenshots, or using the wrong filename, will not be accepted.

## Preparation

Don't forget to follow the assignment submission process described under [GitHub](../../information/github.md) while working on this laboratory.

!!! important "Reviewer"
    When creating the pull request for this laboratory, assign it to the `balazskvancz` GitHub user.

!!! tip "Jupyter notebook or plain Python files"
    You may solve the exercises either in a Jupyter notebook or in plain `.py` files, whichever you prefer. Regardless of the format you choose, make sure that **everything** (code, and the required screenshots) is committed and pushed to your solution branch.

## Setup

!!! important "Mandatory"
    Issue the following command to install the necessary libraries.

    ```bash
    pip install transformers torch torchvision umap-learn
    ```

!!! tip "Optional"
    Install `jupyter` if you are working on your own machine.

    ```bash
    pip install jupyter
    ```

## Introduction to Tokenization

```python
from transformers import AutoTokenizer

# You can specify here which pretrained model you
# would like to use. It is highly recommended to check out
# and experiment with different ones.
#
# Link: https://huggingface.co/google-bert/bert-base-uncased
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

example_input_text = "AI Technologies is a useful and interesting course. I am glad I took it!"

# This step splits the sentence into smaller units called tokens.
tokens = tokenizer.tokenize(example_input_text)

print("Tokens:")
print(tokens)

# Encode the sentence using the tokenizer:
# this converts the tokens into numerical IDs and generates additional information.
encoded = tokenizer(example_input_text)

# The numerical IDs corresponding to each token. These IDs are
# what the neural network actually receives as input.
print("\nToken IDs:")
print(encoded["input_ids"])

# This mask indicates which positions correspond to real tokens – denoted by 1 –
# and which correspond to padding – which is represented with 0.
print("\nAttention mask:")
print(encoded["attention_mask"])
```

## Introduction to Vector Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Loading a pretrained tokenizer and model.
# This model is commonly used for generating sentence embeddings.
#
# Link: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

example_input_text = "AI Technologies is a useful and interesting course. I am glad I took it!"

inputs = tokenizer(
  example_input_text,
  return_tensors="pt" # You can tell the tokenizer to return PyTorch tensors.
)

# The previously tokenized inputs are passed
# through the transformer model, which produces
#  contextual embeddings for each token.
with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state

# Shape looks like the following:
#(batch_size, sequence_length, hidden_dimension)
print("Token embedding tensor shape:")
print(token_embeddings.shape)

# You can compute a sentence embedding using mean pooling.
# This averages the embeddings across the token dimension.
sentence_embedding = token_embeddings.mean(dim=1)

print("\nSentence embedding shape:")
print(sentence_embedding.shape)
```

## Introduction to Measuring Similarity

```python
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# The same model is used as before.
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def calculate_similarity_matrix(words: List[str]):
  inputs = tokenizer(words, padding=True, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)

  token_embeddings  = outputs.last_hidden_state
  word_embeddings   = token_embeddings.mean(dim=1)

  # The scikit-learn library contains utility function
  # to calculate the Cosine similarity, for multiple embeddings.
  similarity_matrix = cosine_similarity(word_embeddings.numpy())

  print(f"The input array: {words}")
  print("Similarity matrix:")
  print(similarity_matrix)
  print("="*50)


similar_words = ["car", "automobile"]
words = ["car", "automobile", "banana", "vehicle"]

calculate_similarity_matrix(similar_words)
calculate_similarity_matrix(words)
```

## Introduction to Dimensionality Reduction and Visualisation

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The AI Technologies university course is helpful.",
    "Learning calculus can be painful.",
    "Artificial intelligence is transforming technology.",
    "Machine learning models require large datasets.",
    "Running or doing some kind of exercise is good for you.",
    "Modern processors are becoming more powerful."
]

# The same process as before, tokenization
# and creating embeddings out of them.
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state

sentence_embeddings = token_embeddings.mean(dim=1)

embeddings = sentence_embeddings.numpy()

# Since these embeddings are in higher dimensions,
# it is quite hard to visualise and understand them.
# We can mitigate this issue, by employing dimension
# reduction techniques. This will transform the vector
# embeddings to 2D space, which can be visualised easier.
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

plt.figure(figsize=(8,6))

for i, sentence in enumerate(sentences):
    x = embedding_2d[i,0]
    y = embedding_2d[i,1]

    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, sentence, fontsize=9)

plt.title("2D Visualisation of Sentence Embeddings")
plt.xlabel("UMAP dimension 1")
plt.ylabel("UMAP dimension 2")
plt.show()
```

## Introduction to Logistic Regression

`Logistic Regression` is a supervised learning algorithm used for classification tasks. Despite its name, it is primarily used to predict discrete class labels, not continuous values. This is not strictly part of the `NLP` process, but you will need it when working on the independent assignments.

The model learns a linear decision boundary by estimating the probability that a given input belongs to a particular class using the logistic (sigmoid) function. For multi-class problems, scikit-learn uses extensions such as one-vs-rest or multinomial logistic regression.

In NLP applications, Logistic Regression is commonly trained on text embeddings to classify sentences or documents into categories, making it a simple yet effective baseline model.

The following code snippet contains a similar example to what you will need later on.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target

# Typical way of splitting the raw
# dataset into training and testing part.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)

# Easy to use function to initialize
# the model via providing the chosen arguments.
#
# The following example uses L2 regularization
# which is also known as Ridge Regression.
model = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    max_iter=1000
)

# From now on, the typical fitting and testing
# of the previously defined model.
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

## Introduction to KMeans

`KMeans` is an unsupervised learning algorithm used to group data points into a predefined number of clusters (k). The algorithm works by iteratively assigning each data point to the nearest cluster centroid and then updating the centroids based on the assigned points. This is not strictly part of the `NLP` process, but you will need it when working on the independent assignments, just as before.

The objective of KMeans is to minimize the within-cluster variance, meaning that points within the same cluster should be as similar as possible.

In the context of NLP, KMeans is often applied to sentence embeddings to discover natural groupings of texts based on semantic similarity, without using any labels.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# For this example, we are going
# to generate dummy dataset, which
# naturally contains three distinguishable
# centers with points around them.
X, _ = make_blobs(
    n_samples=200,
    centers=3,
    n_features=2,
    random_state=42
)

# Sklearn also provides a simple
# way to create and train KMeans-based
# models. You only have to provide the
# desired number of clusters beforehand.
kmeans = KMeans(n_clusters=3, random_state=42)

# And fit the model based on the data.
kmeans.fit(X)

# You can query both the labels and the centers.
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Simple plot, to visualize the results.
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)

plt.title("KMeans Example")
plt.show()
```

## Sentence dataset

The following sentence bank is used as the searchable/clusterable dataset for Exercises 1 and 2.

```python
from typing import List
import random

SENTENCE_BANK = [
  "Artificial intelligence is transforming modern industries.",
  "Machine learning models require large datasets.",
  "Neural networks can recognize complex patterns.",
  "Deep learning is widely used in natural language processing.",
  "Data science combines statistics and programming.",
  "Many companies invest heavily in AI research.",
  "Modern processors are becoming increasingly powerful.",
  "Cloud computing enables scalable applications.",
  "Cybersecurity is essential for protecting sensitive data.",
  "Programming skills are valuable in many careers.",
  "Football is one of the most popular sports in the world.",
  "The team scored two goals in the final match.",
  "Basketball players must be fast and agile.",
  "The athlete trained hard for the competition.",
  "The crowd cheered loudly during the game.",
  "Tennis requires excellent coordination and focus.",
  "The championship game attracted many fans.",
  "Regular exercise improves physical health.",
  "Running is a simple and effective form of exercise.",
  "Many children enjoy playing sports with their friends.",
  "Pizza is a popular Italian dish around the world.",
  "Fresh ingredients make food taste better.",
  "Many people enjoy cooking at home.",
  "Pasta can be prepared in many different ways.",
  "A balanced diet is important for good health.",
  "The restaurant serves delicious traditional meals.",
  "Baking bread requires patience and practice.",
  "Vegetables provide essential nutrients.",
  "Coffee is a common morning beverage.",
  "Chocolate desserts are loved by many people.",
  "Traveling allows people to experience different cultures.",
  "Many tourists visit famous landmarks every year.",
  "Airplanes make long-distance travel faster.",
  "Backpacking is a popular travel style among students.",
  "Hotels provide accommodation for travelers.",
  "Exploring new cities can be exciting.",
  "Maps help travelers navigate unfamiliar places.",
  "Tourists often take photos of historic buildings.",
  "Many people enjoy relaxing vacations by the sea.",
  "Travel blogs share useful travel experiences.",
  "Physics studies the fundamental laws of nature.",
  "Biology examines living organisms and ecosystems.",
  "Chemistry explains how substances interact.",
  "Astronomy explores planets, stars, and galaxies.",
  "Scientists conduct experiments to test hypotheses.",
  "Research advances our understanding of the universe.",
  "Laboratories contain specialized scientific equipment.",
  "Mathematics provides tools for scientific analysis.",
  "Space telescopes capture images of distant galaxies.",
  "Scientific discoveries often lead to new technologies.",
  "Reading books can improve vocabulary and knowledge.",
  "Students attend lectures at the university.",
  "Teachers prepare lessons for their classes.",
  "Education helps people develop critical thinking.",
  "Many students study in the library.",
  "Online courses provide flexible learning opportunities.",
  "Homework helps reinforce new knowledge.",
  "Group projects encourage collaboration among students.",
  "Universities support research and innovation.",
  "Learning new skills takes time and practice.",
  "Music can influence people's emotions.",
  "Many people listen to music while working.",
  "Concerts bring together large audiences.",
  "Learning to play an instrument requires dedication.",
  "Classical music has a long tradition.",
  "Streaming platforms make music widely accessible.",
  "Singers often perform live on stage.",
  "Bands practice regularly before performances.",
  "Music festivals attract thousands of visitors.",
  "Different cultures have unique musical styles.",
  "Morning walks can improve mental health.",
  "Many people drink tea in the afternoon.",
  "Daily routines help organize our time.",
  "Exercise and sleep are important for wellbeing.",
  "People often meet friends at cafes.",
  "Weekend activities help people relax.",
  "Gardening is a peaceful hobby.",
  "Pets can bring joy to their owners.",
  "Many people enjoy watching movies at home.",
  "Good habits contribute to a healthy lifestyle.",
  "Technology companies release new smartphones every year.",
  "Software updates improve system security.",
  "Many developers contribute to open source projects.",
  "Smart devices are becoming more common in homes.",
  "The internet connects people around the world.",
  "Mobile apps simplify everyday tasks.",
  "Digital tools improve productivity.",
  "Online platforms enable remote collaboration.",
  "Artificial intelligence assists medical diagnosis.",
  "Autonomous vehicles are an active research area."
]

def get_random_sentences(n = None) -> List[str]:
  if n is None:
    return SENTENCE_BANK

  if n > len(SENTENCE_BANK):
    raise ValueError("Requested number of sentences exceeds dataset size.")

  return random.sample(SENTENCE_BANK, n)
```

## Exercise 1 – Simple Semantic Similarity Search Engine

Implement a simple semantic similarity search in the function `exercise_one`.

- The function should take the following arguments:
    - `sentences`: a list of sentences that represent the searchable dataset,
    - `query`: a sentence used as the search query,
    - `top_k`: the number of most similar sentences to return.
- Compute sentence embeddings.
- Determine which sentences in the dataset are most semantically similar to the query.
- Filter the `top_k` elements.

```python
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List
import torch

# You can experiment with different
# pretrained models as well.
PRE_TRAINED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def exercise_one(sentences: List[str], query: str, top_k = 3):
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
  # TODO:
  #   - tokenization of the sentences – including the query(!),
  #   - computing embeddings,
  #   - calculating cosine similarity,
  #   - returning the top k entity.
  pass
```

```python
query = "Software development and AI technologies"

exercise_one(SENTENCE_BANK, query, 10)
```

!!! note ":camera: Screenshot"
    Take a screenshot of the output of `exercise_one` and upload it together with your solution as `f1.png`.

## Exercise 2 – Clustering and Visualisation of Sentences

Implement a function called `exercise_2` that takes the following arguments:

- `sentences`: a list of sentences,
- `k`: the number of clusters.

The function should perform the following steps:

- Compute sentence embeddings using a pretrained transformer model.
- Group the sentences into k clusters based on their semantic similarity. Use the `KMeans` model provided by the `scikit-learn` library.
- Apply a dimensionality reduction technique – use `UMAP` – to project the embeddings into a two-dimensional space.
- Create a visualisation where:
    - each point represents a sentence,
    - points are colored according to their assigned cluster.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
from typing import List

PRE_TRAINED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def exercise_2(sentences: List[str], k: int):
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
  # TODO:
  #   - loading the pre-trained model and tokenizer
  #   - computing embeddings,
  #   - perform kmeans clustering,
  #   - dimensionality reduction to 2d,
  #   - plotting based on the created clusters.
  pass
```

```python
sentences = get_random_sentences(20)

exercise_2(sentences, 5)
```

!!! note ":camera: Screenshot"
    Take a screenshot of the resulting cluster visualisation and upload it together with your solution as `f2.png`.

## Exercise 3 – Multi-class News Categorization

Classify news headlines into four categories using embeddings and a simple classifier.

- Load the `AG News dataset` using the `datasets` library. It is sufficient to only load 1000 examples.
- Analyze class balance.
- Use a pretrained transformer to generate sentence embeddings, like `all-MiniLM-L6-v2`.
- Train a `Logistic Regression classifier` – use the one provided by `scikit-learn` – on the embeddings.
- Evaluate using:
    - accuracy,
    - precision,
    - recall,
    - F1-score,
    - confusion matrix on a heatmap.
- Visualise embeddings in 2D – UMAP – with points colored by predicted class.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

PRE_TRAINED_MODEL_NAME=None

def load_ds():
  dataset = load_dataset("ag_news", split="train[:1000]")
  sentences = None # TODO
  labels = None # TODO

  # TODO: Use 20% test split.
  X_train, X_test, y_train, y_test = None, None, None, None

  return X_train, X_test, y_train, y_test

def create_embeddings(model, tokenizer, sentences: List[str]) -> np.ndarray:
  # TODO
  embeddings = None

  return embeddings

def evaluate_results(y_true, y_pred):
  # TODO: classification report

  # TODO: confusion matrix
  # TODO: heatmap based on confusion matrix
  cm = None

def visualise(X_test_emb, y_pred):
  reducer = None
  embedding_2d = reducer.fit_transform(X_test_emb)

  # TODO: plot

def exercise_3():
  X_train, X_test, y_train, y_test = load_ds()

  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  # TODO
  X_train_emb = None
  X_test_emb = None

  # TODO
  y_pred = None

  evaluate_results(y_test, y_pred)
  visualise(X_test_emb, y_pred)
```

```python
exercise_3()
```

!!! note ":camera: Screenshot"
    Take a screenshot of the evaluation metrics, the confusion matrix heatmap, and the UMAP visualisation, and upload it together with your solution as `f3.png`.

## Exercise 4 – Paraphrase Detection Using Sentence Embeddings

Detect whether two sentences express the same thing, meaning using sentence embeddings and similarity scores.

- Load the Quora Question Pairs dataset (`AlekseyKorshuk/quora-question-pairs`) using the datasets library. It is sufficient to only load 1000 examples.
- Extract the needed fields.
- Use a pretrained transformer to generate sentence embeddings.
- Compute the cosine similarity between the embeddings of each sentence pair.
- Convert similarity scores into predictions using a similarity threshold:
    - similarity ≥ threshold → paraphrase,
    - similarity < threshold → not paraphrase.
- Evaluate the results using:
    - accuracy,
    - precision,
    - recall,
    - F1-score.
- Plot the distribution of similarity scores for:
    - paraphrase pairs,
    - non-paraphrase pairs.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import matplotlib.pyplot as plt

PRE_TRAINED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
CLASSIFICATION_THRESHOLD=0.75

def load_ds():
  dataset = load_dataset("AlekseyKorshuk/quora-question-pairs", split="train[:1000]")

  # TODO:
  #   - inspect the loaded dataset,
  #   - separate the questions and labels.
  questions1, questions2, labels = None, None, None

  return questions1, questions2, labels


def create_embedding(model, tokenizer, sentences):
  # TODO: compute vector embeddings.
  embeddings = None

  return embeddings

def visualise_similarity_distribution(similarities, labels):
  # TODO: plot.
  pass

def exercise_4():
  questions1, questions2, labels = load_ds()

  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  # TODO:
  #   - compute vector embeddings,
  #   - calculate cosine similarities,
  #   - classify the pairs based on the threshold value,
  #   - display the classification report,
  #   - plot similarity distribution.

  # TODO:
  # visualise_similarity_distribution(similarities, labels)
```

```python
exercise_4()
```

!!! note ":camera: Screenshot"
    Take a screenshot of the classification report and the similarity distribution plot, and upload it together with your solution as `f4.png`.

## Giving feedback

Please share your opinion about this laboratory. You can include:

- Your general impression of the laboratory.
- The things you liked or did not like.
- Whether you found the lab easy or difficult.
- Which exercises you enjoyed the most.
- What was necessary but not included in this laboratory.
- What was unnecessary but included in this laboratory.
