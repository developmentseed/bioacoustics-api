from django.db import models

from connection import Collection


class MyModel(models.Model):
    name = models.CharField(max_length=100)
    vector = models.BinaryField()

    @staticmethod
    def search(query_vector, top_k=10):
        # Search the Milvus collection for similar vectors
        collection = Collection.get_collection()
        query_data = [float(x) for x in query_vector.split(',')]
        results = collection.search(query_data, top_k=top_k)
        return [MyModel.objects.get(id=r.id) for r in results]

    def as_vector_string(self):
        # Format the vector as a comma-separated string
        return ','.join(str(x) for x in self.vector)
