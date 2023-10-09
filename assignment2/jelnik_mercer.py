def jelinek_mercer(query, documents):
    l = 0.5 # lambda = 0.5
    words_in_all_docs = sum([len(document.split(" ")) for document in documents])
    query_words = query.split(" ")
    weights = []
    for document in documents:
        weight = 1
        for word in query_words:
            words_in_current_doc = len(document.split(" "))
            hits_in_current_doc = document.split(" ").count(word)
            hits_in_all_docs = sum([1 if word in document else 0 for document in documents])
            weight *= (1-l)*(hits_in_current_doc / words_in_current_doc) + l*(hits_in_all_docs / words_in_all_docs)
        weights.append(weight)

    return weights

d1 = "An apple a day keeps the doctor away."
d2 = "The best doctor is the one you run to and can’t find."
d3 = "One rotten apple spoils the whole barrel."
documents=[d1, d2, d3]
q1 = "doctor"
q3 = "doctor apple"

print(jelinek_mercer(q1,documents))
print(jelinek_mercer(q3,documents))