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

"""
First 3 topics:

Topic 0:
0.147*"labour" + 0.137*"price" + 0.128*"employ" + 0.128*"produc" + 0.123*"capit" + 0.122*"countri" + 0.119*"trade" + 0.119*"hi" + 0.115*"tax" + 0.113*"land"

Topic 1:
0.994*"" + 0.074*"0" + 0.021*"1" + 0.021*"2" + 0.020*"barrel" + 0.018*"8" + 0.017*"wheat" + 0.017*"£" + 0.015*"3" + 0.013*"10"

Topic 2:
-0.260*"labour" + 0.202*"silver" + -0.201*"rent" + -0.196*"stock" + 0.192*"gold" + -0.190*"land" + -0.185*"employ" + -0.183*"profit" + -0.175*"capit" + -0.158*"wage"

Comments:

Here, the first topic seems to be about the economy in general, the second topic seems to be about the price of wheat, and the third topic seems to be about the value of gold and silver.
"""

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

"""
TF-IDF weights for query 1:

money: 0.3144551165055443
function: 0.9492723422198104
"""

# 4.3: Calculate document similarity using TF-IDF
doc2similarity1 = enumerate(matrix_similarity_object[tfidf_query1])
doc2similarity2 = enumerate(matrix_similarity_object[tfidf_query2])
top_3_paragraphs = list(sorted(doc2similarity1, key=lambda kv: -kv[1])[:3])

# Print the top 3 paragraphs for query 2
print("Top 3 paragraphs:")
for i, sim in top_3_paragraphs:
    truncated_paragraph = paragraphs[i][:500]
    print(f"[paragraph {i}]\n{truncated_paragraph}\n")

"""
Top 3 paragraphs (Truncated to 500 characters, I previously removed newlines within paragraphs, so I couldn't truncate to 5 lines):

[paragraph 681]
The general stock of any country or society is the same with that of all its inhabitants or members; and, therefore, naturally divides itself into the same three portions, each of which has a distinct function or office.

[paragraph 992]
That wealth consists in money, or in gold and silver, is a popular notion which naturally arises from the double function of money, as the instrument of commerce, and as the measure of value. In consequence of its being the instrument of commerce, when we have money we can more readily obtain whatever else we have occasion for, than by means of any other commodity. The great affair, we always find, is to get money. When that is obtained, there is no difficulty in making any subsequent purchase.

[paragraph 816]
Whatever part of his stock a man employs as a capital, he always expects it to be replaced to him with a profit. He employs it, therefore, in maintaining productive hands only; and after having served in the function of a capital to him, it constitutes a revenue to them. Whenever he employs any part of it in maintaining unproductive hands of any kind, that part is from that moment withdrawn from his capital, and placed in his stock reserved for immediate consumption.

Comments:


"""

# 4.4: Calculate LSI-based topic similarity
lsi_query1 = lsi_model[tfidf_query1]
lsi_query2 = lsi_model[tfidf_query2]

top_3_topics = sorted(lsi_query1, key=lambda kv: -abs(kv[1]))[:3]

doc1similarity_lsi = enumerate(lsi_index[lsi_query1])
top_3_paragraphs = sorted(doc1similarity_lsi, key=lambda kv: -kv[1])[:3]

print("Top 3 lsi topics:")
for i, u in top_3_topics:
    topic = lsi_model.show_topic(i)
    print(f"[topic {i}]\n{topic}\n")

"""
Top 3 lsi topics:

[topic 5]
[('bank', -0.2602332606912621), ('price', 0.23692618250902694), ('circul', -0.23096142234591904), ('capit', -0.22646285079405615), ('gold', -0.19252382597407494), ('money', -0.18599007130386283), ('corn', 0.1673125131684517), ('tax', 0.14946201834559633), ('coin', -0.14135900461018128), ('silver', -0.13716795091335146)]

[topic 16]
[('coloni', -0.3098670913834844), ('circul', -0.2754030645588527), ('price', -0.1891565783232087), ('increas', 0.1686980471230326), ('money', -0.15301517387567876), ('trade', 0.14569436840302694), ('coin', 0.13882312166439936), ('cent', 0.1384186865396116), ('per', 0.134419167323987), ('materi', -0.13381053667835402)]

[topic 63]
[('proport', 0.23569871168742196), ('money', -0.13459923766052662), ('scarciti', -0.12604449752574512), ('provis', -0.11627310079382817), ('improv', 0.11312791686710019), ('rise', -0.11294207510279232), ('year', -0.11168736554712913), ('between', 0.11145217314394722), ('interest', 0.11075121042404544), ('silver', 0.11031708256120382)]

Comments:

The first topic seems to be about the banking system, the second topic seems to be about colonialism, and the third topic seems to be about scarcity and supply.

I however don't always get the same topics. Instead of topic 63, sometimes when running the script I get:

[topic 56]
[('parish', 0.1693757616188269), ('hous', -0.1515699884729931), ('money', 0.15083051264264455), ('demand', -0.14796090637765721), ('wool', -0.11921743776026561), ('consum', -0.1152545814913234), ('bill', -0.11125838920592619), ('import', 0.10509568793738766), ('ii', 0.10407993663601532), ('profit', 0.10341291970009894)]

This topic seems to be about the housing market.
"""


print("Top 3 lsi paragraphs:")
for i, u in top_3_paragraphs:
    truncated_paragraph = paragraphs[i][:500]
    print(f"[paragraph {i}]\n{truncated_paragraph}\n")

"""
Top 3 lsi paragraphs (Truncated to 500 characters, I previously removed newlines within paragraphs, so I couldn't truncate to 5 lines):

Top 3 lsi paragraphs:
[paragraph 992]
That wealth consists in money, or in gold and silver, is a popular notion which naturally arises from the double function of money, as the instrument of commerce, and as the measure of value. In consequence of its being the instrument of commerce, when we have money we can more readily obtain whatever else we have occasion for, than by means of any other commodity. The great affair, we always find, is to get money. When that is obtained, there is no difficulty in making any subsequent purchase.

[paragraph 1007]
No complaint, however, is more common than that of a scarcity of money. Money, like wine, must always be scarce with those who have neither wherewithal to buy it, nor credit to borrow it. Those who have either, will seldom be in want either of the money, or of the wine which they have occasion for. This complaint, however, of the scarcity of money, is not always confined to improvident spendthrifts. It is sometimes general through a whole mercantile town and the country in its neighbourhood. Ove

[paragraph 1008]
It would be too ridiculous to go about seriously to prove, that wealth does not consist in money, or in gold and silver; but in what money purchases, and is valuable only for purchasing. Money, no doubt, makes always a part of the national capital; but it has already been shown that it generally makes but a small part, and always the most unprofitable part of it.

Comparison to the paragraphs found with the TD IDF model:

The paragraphs found with the LSI model are very similar to the paragraphs found with the TF IDF model. 
The first paragraph is the same, and the second and third paragraphs are very similar. 
The LSI model seems to be better at finding similar paragraphs than the TF IDF model, but the TF IDF model seems to be better at finding the most similar paragraphs.
"""