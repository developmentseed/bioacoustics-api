from urllib.parse import urljoin
import faiss
import json
from numpy import array
import requests

from django.conf import settings
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework import serializers
from drf_spectacular.utils import extend_schema, extend_schema_field
from drf_spectacular.types import OpenApiTypes

from .connection import MilvusConnection

pca_matrix = faiss.read_VectorTransform('./bioacoustics/milvus/1280_to_256_dim_redux_combined_only.pca')


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
    audio_url = serializers.SerializerMethodField()

    @extend_schema_field(OpenApiTypes.STR)
    def get_image_url(self, obj):
        return self.format_media_url(obj, 'png')

    @extend_schema_field(OpenApiTypes.STR)
    def get_audio_url(self, obj):
        return self.format_media_url(obj, 'flac')

    def format_media_url(self, obj, file_format):
        return urljoin(
            settings.A2O_API_URL,
            'audio_recordings/{}/media.{}?start_offset={}&end_offset={}'.format(
                obj.file_seq_id,
                file_format,
                obj.clip_offset_in_file,
                obj.clip_offset_in_file + 5
            )
        )


class ResultSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    distance = serializers.FloatField()
    entity = EntitySerializer()


class EmbedSerializer(serializers.Serializer):
    audio_file = serializers.FileField()


class EmbedResultSerializer(serializers.Serializer):
    embedding = serializers.ListField()


class SearchSerializer(serializers.Serializer):
    audio_file = serializers.FileField(required=False, allow_null=True)
    embed = serializers.CharField(required=False, allow_null=True)
    metric_type = serializers.CharField(required=False, allow_null=True)
    nprobe = serializers.IntegerField(required=False, allow_null=True)
    limit = serializers.IntegerField(
        min_value=1,
        max_value=16384,
        required=False,
        allow_null=True
    )
    offset = serializers.IntegerField(
        required=False,
        allow_null=True
    )
    expression = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True
    )

    def validate(self, data):
        if data['embed'] or data['audio_file']:
            return data
        else:
            raise serializers.ValidationError(
                'embed or audio_file fields can not be both null.'
            )

@extend_schema(
    request=SearchSerializer,
    responses={200: ResultSerializer(many=True)},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@parser_classes([FormParser, MultiPartParser])
def search_view(request):
    """
    Executes a search in the Milvus Database using an audio file or an embedding as input.
    """
    data = {
        'audio_file': request.FILES.get('audio_file'),
        'embed': request.data.get('embed'),
        'limit': request.data.get('limit'),
        'offset': request.data.get('offset'),
        'expression': request.data.get('expression'),
        'metric_type': request.data.get('metric_type'),
        'nprobe': request.data.get('nprobe')
    }
    search = SearchSerializer(data=data)

    if not search.is_valid():
        return Response(search.errors, 400)

    search_params = {**search.validated_data}

    if search_params.get('embed') is None:
        status_code, embed_data = get_embedding(
            search_params.get('audio_file')
        )
        embed = embed_data['embedding']

        if status_code != 200:
            return Response(embed_data, 400)
    else:
        embed = json.loads(search_params.get('embed'))

    m = MilvusConnection()
    search_vector = pca_transform(embed)
    results = m.search(
        search_vector,
        expression=search_params.get('expression'),
        limit=search_params.get('limit'),
        offset=search_params.get('offset'),
        metric_type=search_params.get('metric_type') or 'IP',
        nprobe=search_params.get('nprobe') or 16
    )
    serializer = ResultSerializer(results[0], many=True)

    return Response(serializer.data)


@extend_schema(
    request=EmbedSerializer(),
    responses={200: EmbedResultSerializer},
)
@api_view(['POST'])
@permission_classes([AllowAny])
@parser_classes([FormParser, MultiPartParser])
def embed_view(request):
    """Get the embedding array for an audio file. Audio length must be 5 seconds."""
    serializer = EmbedSerializer(data={
        'audio_file': request.FILES.get('audio_file'),
    })

    if not serializer.is_valid():
        return Response(serializer.errors, 400)

    validated_data = {**serializer.validated_data}
    status_code, embed_data = get_embedding(
        validated_data.get('audio_file')
    )

    return Response(embed_data, status_code)


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
