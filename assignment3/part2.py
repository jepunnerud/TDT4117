import random; random.seed(123)
import codecs
import string
import gensim
from nltk.stem.porter import PorterStemmer

def make_paragraphs(original_list):
    result = []
    current_sublist = []
    for sublist in original_list:
        if sublist == ['']:
            if current_sublist:
                result.append(current_sublist)
            current_sublist = []
        else:
            current_sublist.extend(sublist)
    if current_sublist:
        result.append(current_sublist)
    return result

try:
    f = codecs.open("pg3300.txt", "r", "utf-8")
except IOError:
    print("Could not open file")
    exit()

text = f.read().splitlines()

try:
    g = codecs.open("stopwords.txt", "r", "utf-8")
except IOError:
    print("Could not open file")
    exit()

stopwords = g.read().split(",")

def partition_into_paragraphs(file_path):
    paragraphs = []
    with codecs.open(file_path, 'r', 'utf-8') as file:
        current_paragraph = []
        for line in file:
            # If the line is empty or contains only whitespace (indicating a new paragraph), append the current paragraph and start a new one.
            if line.strip() == '':
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line.strip())

        # Append the last paragraph (if any) after reaching the end of the file.
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs

paragraphs = partition_into_paragraphs("pg3300.txt")
text = paragraphs
# print(text[:10])

text = list(filter(lambda x: not "Gutenberg" in x, text)) # Remove Gutenberg header
text = [x.strip().split(" ") for x in text] # Split into words
# text = [x for x in text if x != ['']] # Remove empty lines
# text = make_paragraphs(text) # Split into paragraphs

print(text[:10])

def remove_punctuation(word):
    translator = str.maketrans('', '', string.punctuation)
    return word.translate(translator)

for i, sentence in enumerate(text):
    text[i] = [remove_punctuation(word) for word in sentence] # Remove punctuation

stemmer = PorterStemmer()
for i, sentence in enumerate(text):
    text[i] = [stemmer.stem(word.lower()) for word in sentence] # Stem words

# print(text[:10])

dictionary = gensim.corpora.Dictionary(text)

stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)
for sentence in text:
    dictionary.doc2bow(sentence, allow_update=True)

# print(dictionary)

tfidf_model = gensim.models.TfidfModel(dictionary=dictionary)

tfidf_corpus = tfidf_model[dictionary]

matrix_similarity_object = gensim.similarities.MatrixSimilarity([tfidf_corpus], num_features=len(dictionary))

lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)

lsi_corpus = lsi_model[tfidf_corpus]

lsi_model.show_topic()

f.close