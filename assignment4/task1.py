from codecs import open
import string
import re

text = "Intelligent behavior in people is a product of the mind. But the mind itself is more like what the human brain does."

d1 = "Although we know much more about the human brain than we did even"
d2 = "ten years ago, the thinking it engages in remain pretty much a total"
d3 = "mystery. It is like a big jigsaw puzzle where we can see many of the"
d4 = "pieces, but cannot yet put them all together. There is so much about us"
d5 = "that we do not understand at all."

docs = [d1, d2, d3, d4, d5]

with open("stopwords.txt", "r", "utf-8") as g:
    stopwords = g.read().split(",")

text_lower_no_punctuation = text.lower().translate(str.maketrans("", "", string.punctuation))

# a

indexed_words = {word: [t.start() for t in re.finditer(word.lower(), text_lower_no_punctuation)] for word in text_lower_no_punctuation.split() if word not in stopwords}

# print(indexed_words)

# b

blocks = [text_lower_no_punctuation.split()[i:i+4] for i in range(0, len(text_lower_no_punctuation.split()), 4)]

for i in range(len(blocks)):
    # print("Block " + str(i+1) + ": " + " ".join(blocks[i]))
    pass

indexed_words_with_blocks = {}
for block in blocks:
    for word in block:
        if word not in stopwords:
            if word not in indexed_words_with_blocks:
                indexed_words_with_blocks[word] = [1, [blocks.index(block)+1]]
            else:
                indexed_words_with_blocks[word][1].append(blocks.index(block)+1)
                indexed_words_with_blocks[word][0] += 1

for word in indexed_words_with_blocks:
    # print(word + ": " + str(indexed_words_with_blocks[word]))
    pass
# d

def make_blocks_multiple_documents(docs):
    docs_lower_no_punctuation = [doc.lower().translate(str.maketrans("", "", string.punctuation)) for doc in docs]
    vocabulary = [word for word in " ".join(docs_lower_no_punctuation).split() if word not in stopwords]
    indexed_words_with_blocks = {word: [] for word in vocabulary}
    docs_in_blocks = [[doc.split()[i:i+4] for i in range(0, len(doc.split()), 4)] for doc in docs_lower_no_punctuation]

    for word in vocabulary:
        for i, doc in enumerate(docs_in_blocks):
            occurrences = []
            for j, block in enumerate(doc):
                if word in block:
                    occurrences.append(j + 1)
            if occurrences and (i + 1, len(occurrences), tuple(occurrences)) not in indexed_words_with_blocks[word]:
                indexed_words_with_blocks[word].append(
                    (i + 1, len(occurrences), tuple(occurrences))
                )
    for k, v in indexed_words_with_blocks.items():
        indexed_words_with_blocks[k] = tuple(v)
    return indexed_words_with_blocks

print(make_blocks_multiple_documents(docs))