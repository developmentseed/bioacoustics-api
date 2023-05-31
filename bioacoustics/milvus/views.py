import requests

from django.conf import settings
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from rest_framework import serializers

from .connection import MilvusConnection


class EntitySerializer(serializers.Serializer):
    site_id = serializers.IntegerField()
    site_name = serializers.CharField()
    subsite_name = serializers.CharField()
    filename = serializers.CharField()
    file_seq_id = serializers.CharField()
    file_timestamp = serializers.IntegerField()
    offset = serializers.IntegerField()


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
    results = m.search(
        embed_data['embedding'],
        search_params.get('expression'),
        search_params.get('limit')
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
