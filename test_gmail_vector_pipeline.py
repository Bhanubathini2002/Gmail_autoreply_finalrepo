# test_gmail_vector_pipeline.py
from embedder import OllamaEmbedder
from vector_store import GmailVectorStore

embedder = OllamaEmbedder()
text1 = "Subject: Meeting update\nBody: The client meeting is moved to 4 PM."
text2 = "Subject: Invoice reminder\nBody: Please send the invoice before Friday."
text3 = "Subject: Dinner plan\nBody: Let's go out for dinner tonight."

# Generate embeddings
vec1 = embedder.embed(text1)
vec2 = embedder.embed(text2)
vec3 = embedder.embed(text3)

store = GmailVectorStore(dim=len(vec1))

# Insert into Milvus
store.insert_email("Meeting update", "team@company.com", text1, vec1)
store.insert_email("Invoice reminder", "accounts@company.com", text2, vec2)
store.insert_email("Dinner plan", "friend@mail.com", text3, vec3)

# Search similar to a new query
query = "Meeting rescheduled to 4 PM."
qvec = embedder.embed(query)
results = store.search_similar(qvec)

print("\n🔍 Similar Emails:")
for hit in results:
    subj = hit.entity.get("subject")
    score = hit.distance
    print(f"→ {subj}  (similarity: {1 - score:.3f})")
