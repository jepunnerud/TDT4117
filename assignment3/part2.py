# Import necessary libraries and modules
import random; random.seed(123)
import codecs
import string
import gensim
from nltk.stem.porter import PorterStemmer

# Part 1: Data Preparation

# Read a list of stopwords from a file
with codecs.open("stopwords.txt", "r", "utf-8") as g:
    stopwords = g.read().split(",")

# Function to partition a text file into paragraphs
def partition_into_paragraphs(file_path):
    """
    Partition a text file into paragraphs.

    Args:
        file_path (str): The path to the text file to be processed.

    Returns:
        list: A list of paragraphs, where each paragraph is represented as a string.
    """
    paragraphs = []
    with codecs.open(file_path, 'r', 'utf-8') as file:
        current_paragraph = []
        for line in file:
            if line.strip() == '':
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line.strip())
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
    return paragraphs

# Read and partition a text file into paragraphs
paragraphs = partition_into_paragraphs("pg3300.txt")

# Remove paragraphs containing the word "gutenberg"
paragraphs = list(filter(lambda x: not "gutenberg" in x.lower(), paragraphs))

# Copy the list of paragraphs to the variable 'text'
text = paragraphs[:]

# Tokenize the text by splitting it into words
text = [x.strip().split(" ") for x in text]

# Function to remove punctuation from a word
def remove_punctuation(word):
    """
    Remove punctuation from a word.

    Args:
        word (str): The word to process.

    Returns:
        str: The word with punctuation removed.
    """
    translator = str.maketrans('', '', string.punctuation)
    return word.translate(translator)

# Remove punctuation from each word in the text
for i, sentence in enumerate(text):
    text[i] = [remove_punctuation(word) for word in sentence]

# Initialize a Porter Stemmer for word stemming
stemmer = PorterStemmer()

# Stem and convert words to lowercase in the text
for i, sentence in enumerate(text):
    text[i] = [stemmer.stem(word.lower()) for word in sentence] # Stem words

# Part 2: Create a Gensim Dictionary and Corpus

# Create a Gensim Dictionary from the preprocessed text
dictionary = gensim.corpora.Dictionary(text)

# Filter out stopwords from the dictionary
stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

# Create a Gensim corpus from the preprocessed text
corpus = []

for paragraph in text:
    corpus.append(dictionary.doc2bow(paragraph))

# Part 3: Topic Modeling with LSI

# Create a TF-IDF model from the corpus
tfidf_model = gensim.models.TfidfModel(corpus)

# Transform the corpus to a TF-IDF representation
tfidf_corpus = tfidf_model[corpus]

# Create a Matrix Similarity object for document similarity
matrix_similarity_object = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# Create an LSI model with 100 topics
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)

# Transform the TF-IDF corpus into LSI space
lsi_corpus = lsi_model[tfidf_corpus]

# Create a Matrix Similarity object for LSI document similarity
lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus)

# Show the top 3 topics from the LSI model
topics = lsi_model.show_topics(num_topics=100)

for topic_id, topic in topics[:3]:
    print(f"Topic {topic_id}:\n{topic}\n")

# Part 4: Query Processing and Similarity

# 4.1: Preprocess and tokenize queries
def preprocessing(query):
    """
    Preprocess and tokenize a query.

    Args:
        query (str): The query text.

    Returns:
        list: A list of preprocessed and tokenized words.
    """
    words = query.split()
    words = [remove_punctuation(word) for word in words]
    words = [stemmer.stem(word.lower()) for word in words]
    words = [word for word in words if word not in stopwords]
    return words

query1 = "What is the function of money?"
query2 = "How taxes influence Economics?"

query1_processed = preprocessing(query1)
query2_processed = preprocessing(query2)

# 4.2: Convert queries to TF-IDF representation
query1_bow = dictionary.doc2bow(query1_processed)
query2_bow = dictionary.doc2bow(query2_processed)

tfidf_query1 = tfidf_model[query1_bow]
tfidf_query2 = tfidf_model[query2_bow]

# Print TF-IDF weights for query 1
print("TF-IDF weights for query 1:")
for word_id, weight in tfidf_query1:
    word = dictionary[word_id]
    print(f"{word}: {weight}")
print("\n")

# 4.3: Calculate document similarity using TF-IDF
doc2similarity1 = enumerate(matrix_similarity_object[tfidf_query1])
doc2similarity2 = enumerate(matrix_similarity_object[tfidf_query2])
top_3_paragraphs = list(sorted(doc2similarity2, key=lambda kv: -kv[1])[:3])

# Print the top 3 paragraphs for query 2
print("Top 3 paragraphs:")
for i, sim in top_3_paragraphs:
    paragraph = paragraphs[i].split('\n')[:5]
    truncated_paragraph = "\n".join(paragraph)
    print(f"[paragraph {i}]\n{truncated_paragraph}\n")

# 4.4: Calculate LSI-based topic similarity
lsi_query1 = lsi_model[tfidf_query1]
lsi_query2 = lsi_model[tfidf_query2]

top_3_topics = sorted(lsi_query2, key=lambda kv: -abs(kv[1]))[:3]

doc2similarity_lsi = enumerate(lsi_index[lsi_query2])
top_3_paragraphs = sorted(doc2similarity_lsi, key=lambda kv: -kv[1])[:3]

print("Top 3 lsi topics:")
for i, weight in top_3_topics:
    topic = lsi_model.show_topic(i)
    print(f"[topic {i}]\n{topic}\n")

print("Top 3 lsi paragraphs:")
for i, sim in top_3_paragraphs:
    paragraph = paragraphs[i].split('\n')[:5]
    truncated_paragraph = "\n".join(paragraph)
    print(f"[paragraph {i}]\n{truncated_paragraph}\n")
