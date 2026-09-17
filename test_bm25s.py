import bm25s
corpus = ["hello world", "hello friend"]
tokenizer = bm25s.tokenize(corpus, show_progress=False)
retriever = bm25s.BM25()
retriever.index(tokenizer)
query = "hello"
q_tokens = bm25s.tokenize(query, show_progress=False)
results, scores = retriever.retrieve(q_tokens, k=1, show_progress=False)
print(results)
