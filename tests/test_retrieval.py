from rag.retriever import retrieve 

query = "Explain self-attention in simple terms"
results = retrieve(query, top_k=5)

for i, r in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(r.get("source", ""))
    print(r.get("text", "")[:500])