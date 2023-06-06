from pymilvus import connections, Collection

from django.conf import settings


class MilvusConnection:
    def __init__(self):
        self.connect()
        self.collection = self.get_main_collection()
        self.collection.load()

    def connect(self):
        MILVUS = settings.MILVUS
        connections.connect(
            # alias=MILVUS['DB_ALIAS'],
            # user=MILVUS['DB_USER'],
            # password=MILVUS['DB_PASSWORD'],
            host=MILVUS['DB_HOST'],
            port=MILVUS['DB_PORT'],
        )

    def get_collection(self, collection_name):
        return Collection(name=collection_name)

    def get_main_collection(self):
        return Collection(name=settings.MILVUS['DB_COLLECTION'])

    def drop_index(self, collection_name):
        self.get_collection(collection_name).drop_index()

    def create_index(self, field, index_params):
        self.collection.create_index(field, index_params)
        print(f"Created index {self.collection.index().params}")

    def search(self, query_vector, expression=None, limit=100, offset=0):
        """Search the Milvus collection for similar vectors"""
        search_params = {
            "data": query_vector,
            "anns_field": "embedding",
            "param": {"metric_type": "L2", "params": {"nprobe": 16}},
            "limit": limit if limit else 100,
            "offset": offset if offset else 0,
            "output_fields": [
                "site_id",
                "site_name",
                "subsite_name",
                "file_timestamp",
                "file_seconds_since_midnight",
                "filename",
                "file_seq_id",
                "clip_offset_in_file"
            ]
        }
        if expression:
            search_params['expr'] = expression

        results = self.collection.search(**search_params)
        return results
