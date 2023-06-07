from urllib.parse import urljoin
import faiss
from numpy import array
import requests

from django.conf import settings
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from rest_framework import serializers

from .connection import MilvusConnection

pca_matrix = faiss.read_VectorTransform('./bioacoustics/milvus/1280_to_256_dimensionality_reduction.pca')


class EntitySerializer(serializers.Serializer):
    site_id = serializers.IntegerField()
    site_name = serializers.CharField()
    subsite_name = serializers.CharField()
    filename = serializers.CharField()
    file_seq_id = serializers.CharField()
    file_timestamp = serializers.IntegerField()
    file_seconds_since_midnight = serializers.IntegerField()
    clip_offset_in_file = serializers.IntegerField()
    image_url = serializers.SerializerMethodField()

    def get_image_url(self, obj):
        return urljoin(
            settings.A2O_API_URL,
            'audio_recordings/{}/media.png?start_offset={}&end_offset={}'.format(
                obj.file_seq_id,
                obj.clip_offset_in_file,
                obj.clip_offset_in_file + 5
            )
        )


class ResultSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    distance = serializers.FloatField()
    entity = EntitySerializer()


class SearchSerializer(serializers.Serializer):
    audio_file = serializers.FileField()
    limit = serializers.IntegerField(
        min_value=1,
        max_value=5000,
        required=False,
        allow_null=True
    )
    offset = serializers.IntegerField(
        min_value=0,
        max_value=5000,
        required=False,
        allow_null=True
    )
    expression = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True
    )


@api_view(['POST'])
@permission_classes([AllowAny])
def search_view(request):
    """Executes a search in the Milvus Database using an audio file as input."""
    data = {
        'audio_file': request.FILES.get('audio_file'),
        'limit': request.data.get('limit'),
        'offset': request.data.get('offset'),
        'expression': request.data.get('expression')
    }
    search = SearchSerializer(data=data)

    if not search.is_valid():
        return Response(search.errors, 400)

    search_params = {**search.validated_data}
    status_code, embed_data = get_embedding(
        search_params.get('audio_file')
    )

    if status_code != 200:
        return Response(embed_data, 400)

    m = MilvusConnection()
    search_vector = pca_transform(embed_data['embedding'])
    results = m.search(
        search_vector,
        search_params.get('expression'),
        search_params.get('limit'),
        search_params.get('offset')
    )
    serializer = ResultSerializer(results[0], many=True)

    return Response(serializer.data)


def get_embedding(audio_file):
    """Make a request to the Embed Service URL and return the JSON response."""
    files = {'audio_file': audio_file}
    req = requests.post(
        settings.EMBED_SERVICE_URL,
        files=files,
    )
    return [req.status_code, req.json()]


def pca_transform(embedding):
    """Tranform the search vector from 1280 to 256 dimension numpy array."""
    embed_np = array(embedding).astype('float32')
    reduced_search_vectors = pca_matrix.apply(embed_np[0])
    return reduced_search_vectors
