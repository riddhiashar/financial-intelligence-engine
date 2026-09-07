import numpy as np
import json

from vector_db import (
    get_client,
    create_collection,
    upload_embeddings
)

client = get_client(host="localhost", port=6333)

create_collection(client)

embeddings = np.load("headline_embeddings.npy")
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

upload_embeddings(client, embeddings, metadata)

print("Done - 9380 points uploaded")