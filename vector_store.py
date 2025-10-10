# vector_store.py
from typing import List
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

COLLECTION = "gmail_emails"

class GmailVectorStore:
    def __init__(self, dim: int = 768):
        connections.connect("default", host="127.0.0.1", port="19530")
        if not utility.has_collection(COLLECTION):
            self._create_collection(dim)
        self.col = Collection(COLLECTION)

    def _create_collection(self, dim: int):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="from_email", dtype=DataType.VARCHAR, max_length=320),
            FieldSchema(name="body", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="Gmail emails with embeddings")
        col = Collection(COLLECTION, schema)
        col.create_index(field_name="embedding",
                         index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"})
        print("✅ Created Milvus collection:", COLLECTION)

    def insert_email(self, subject: str, from_email: str, body: str, embedding: List[float]):
        col = Collection(COLLECTION)
        col.insert([[subject], [from_email], [body], [embedding]])
        col.flush()
        print(f"📥 Inserted email: {subject[:50]}...")

    def search_similar(self, query_embedding: List[float], limit: int = 3):
        col = Collection(COLLECTION)
        col.load()
        res = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["subject", "from_email", "body"],
        )
        return res[0]
