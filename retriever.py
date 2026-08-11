"""
RAG MVP
retriever.py

Responsible for

1. Load embedding model
2. Load FAISS index
3. Load metadata
4. Build retrieval query
5. Search vector database
6. Return relevant SOP chunks


Instead it searches using

Intent
+
Current customer context

"""

import os

os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)

import faiss
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

EMBEDDING_MODEL = "all-mpnet-base-v2"

INDEX_PATH = "vector_db/faiss.index"

METADATA_PATH = "vector_db/metadata.parquet"

TOP_K = 3


# ---------------------------------------------------
# RETRIEVER
# ---------------------------------------------------

class Retriever:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Embedding model loaded.")

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            INDEX_PATH
        )

        print("FAISS index loaded.")

        print("Loading metadata...")

        self.metadata = pd.read_parquet(
            METADATA_PATH
        )

        print(f"{len(self.metadata)} chunks loaded.")


    # ------------------------------------------------

    def build_query(
        self,
        intent,
        transcript
    ):
        """
        

        Use the FULL request.

        Not just

            refund

        But

            Refund

            Customer wants a refund for
            a Samsung washing machine.
        """

        return f"""
Intent:
{intent}

Customer Context:
{transcript}
"""


    # ------------------------------------------------

    def embed(
        self,
        text
    ):

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True
        )

        return embedding.astype("float32")


    # ------------------------------------------------

    def search(
        self,
        embedding,
        top_k=TOP_K
    ):

        distances, indices = self.index.search(
            embedding,
            top_k
        )

        return distances[0], indices[0]


    # ------------------------------------------------

    def retrieve(
        self,
        intent,
        transcript,
        top_k=TOP_K
    ):

        query = self.build_query(
            intent,
            transcript
        )

        embedding = self.embed(query)

        distances, indices = self.search(
            embedding,
            top_k
        )

        results = []

        for distance, idx in zip(
            distances,
            indices
        ):

            if idx == -1:
                continue

            row = self.metadata.iloc[idx]

            results.append({

                "intent": intent,

                "score": float(distance),

                "document": row["filename"],

                "chunk_id": row["chunk_id"],

                "chunk_index": int(row["chunk_index"]),

                "text": row["chunk_text"]

            })

        return results


# ---------------------------------------------------
# TEST
# ---------------------------------------------------

if __name__ == "__main__":

    retriever = Retriever()

    transcript = """
Customer says they bought
a Samsung washing machine.

It stopped working.

The customer wants a refund.
"""

    intent = "Refund"

    results = retriever.retrieve(

        intent=intent,

        transcript=transcript

    )

    print()

    print("=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for i, r in enumerate(results, start=1):

        print()

        print(f"Result {i}")

        print("-" * 80)

        print(f"Intent   : {r['intent']}")
        print(f"Document : {r['document']}")
        print(f"Chunk    : {r['chunk_index']}")
        print(f"Score    : {r['score']:.3f}")

        print()

        print(r["text"][:700])