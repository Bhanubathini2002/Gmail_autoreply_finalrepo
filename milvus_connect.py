# milvus_connect.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 1️⃣ Connect to Milvus first
connections.connect("default", host="127.0.0.1", port="19530")
print("✅ Connected to Milvus")

# 2️⃣ (Optional) Drop existing collection if it exists
if utility.has_collection("gmail_test"):
    utility.drop_collection("gmail_test")
    print("🧹 Old collection 'gmail_test' dropped")

# 3️⃣ Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
]
schema = CollectionSchema(fields, description="Test collection for Gmail assistant")

# 4️⃣ Create collection
col = Collection("gmail_test", schema)
print("✅ Created collection 'gmail_test'")

# 5️⃣ Insert sample data
data = [
    ["this is a test email body", "this is another email"],
    [[0.1, 0.2, 0.3, 0.4], [0.9, 0.1, 0.2, 0.3]],
]
col.insert(data)
print("✅ Inserted 2 vectors")

# 6️⃣ Create index before loading
col.create_index(
    field_name="embedding",
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
)
print("✅ Index created")

# 7️⃣ Load and search
col.load()
res = col.search(
    data=[[0.1, 0.2, 0.3, 0.4]],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=2,
    output_fields=["text"],
)
print("🔍 Search results:")
for hit in res[0]:
    print(f"  → score={hit.distance:.4f}, text={hit.entity.get('text')}")
