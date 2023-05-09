from pymilvus import connections, Collection

from django.conf import settings


class Connection:
    def connect(self):
        MILVUS = settings.MILVUS
        connections.connect(
            alias=MILVUS.DB_ALIAS,
            user=MILVUS.DB_USER,
            password=MILVUS.DB_PASSWORD,
            host=MILVUS.DB_HOST,
            port=MILVUS.DB_PORT,
        )

    def get_collection(self, collection_name):
        return Collection(name=collection_name)
