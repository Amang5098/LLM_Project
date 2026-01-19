from sentence_transformers import CrossEncoder

class Search:
    def __init__(self, collection):
        self.collection = collection
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search(self, query: str, top_k_retrieval=20, top_k_rerank=5):

        # Stage 1: Semantic Retrieval (Bio-Encoder)
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k_retrieval
        )
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        # Stage 2: Re-ranking (Cross-Encoder)
        # Prepare pairs: (Query, Document_Context)
        pairs = [[query, doc] for doc in documents]

        # Predict scores
        scores = self.cross_encoder.predict(pairs)

        # Sort by score (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Stage 3: Return top-k results
        retrieved = []
        for rank, idx in enumerate(ranked_indices[:top_k_rerank]):
            retrieved.append({
                "rank": rank+1,
                "score": scores[idx],
                "source": metadatas[idx]['source'],
                "page": metadatas[idx]['page'],
                "content": documents[idx]
            })

        return retrieved

    @classmethod
    def get(cls, collection):
        instance = cls(collection)
        return instance.search