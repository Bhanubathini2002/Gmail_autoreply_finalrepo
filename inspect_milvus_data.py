# inspect_milvus_data.py
from pymilvus import connections, Collection

# Step 1: Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")
print("✅ Connected to Milvus")

# Step 2: Load your collection
collection_name = "gmail_emails"
col = Collection(collection_name)
col.load()

# Step 3: Get collection stats
stats = col.num_entities
print(f"📊 Total entities (rows) stored: {stats}")

# Step 4: Optionally show one or two entries
print("\n🔍 Fetching a few stored emails...")
data = col.query(expr="", output_fields=["subject", "from_email", "body"], limit=3)

for i, doc in enumerate(data, start=1):
    print(f"\n📧 Email #{i}")
    print(f"  Subject: {doc['subject']}")
    print(f"  From: {doc['from_email']}")
    print(f"  Body: {doc['body'][:200]}...")  # show first 200 chars only
