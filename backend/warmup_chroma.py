"""Pre-cache ChromaDB embedding model at Docker build time.
Run once during build to avoid 15-min download on first runtime call."""

from chromadb import Client

c = Client()
col = c.get_or_create_collection("_warmup")
col.add(documents=["warmup"], ids=["warmup"])
col.delete(ids=["warmup"])
c.delete_collection("_warmup")
print("ChromaDB ONNX model cached successfully")
