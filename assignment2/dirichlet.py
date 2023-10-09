def dirichlet(query, documents):
    mu = 8
    words_in_all_docs = sum([len(document.split(" ")) for document in documents])
    query_words = query.split(" ")
    weights = []
    for document in documents:
        weight = 1
        for word in query_words:
            words_in_current_doc = len(document.split(" "))
            hits_in_current_doc = document.split(" ").count(word)
            hits_in_all_docs = sum([1 if word in document else 0 for document in documents])
            weight *= ((hits_in_current_doc / words_in_current_doc) + ((mu * hits_in_all_docs) / words_in_all_docs)) / (mu + words_in_current_doc)
        weights.append(weight)

    return weights



documents = []

d1 = "An apple a day keeps the doctor away."
d2 = "The best doctor is the one you run to and can’t find."
d3 = "One rotten apple spoils the whole barrel."

q1 = "doctor"
q3 = "doctor apple"

documents = [d1, d2, d3]

print(dirichlet(q1, documents))
print(dirichlet(q3, documents)) 