from urllib.parse import urljoin
import secrets
import string
from datetime import datetime
import faiss
import json
from numpy import array
import requests
from minio.error import S3Error

from django.conf import settings
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework import serializers
from drf_spectacular.utils import extend_schema, extend_schema_field
from drf_spectacular.types import OpenApiTypes

from .connection import MilvusConnection
from .file_uploader import upload_file


pca_matrix = faiss.read_VectorTransform('./bioacoustics/milvus/1280_to_256_dimensionality_reduction_filtered.pca')


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
        return self.format_media_url(obj, "png")

    @extend_schema_field(OpenApiTypes.STR)
    def get_audio_url(self, obj):
        return self.format_media_url(obj, "flac")

    def format_media_url(self, obj, file_format):
        return urljoin(
            settings.A2O_API_URL,
            "audio_recordings/{}/media.{}?start_offset={}&end_offset={}".format(
                obj.file_seq_id,
                file_format,
                obj.clip_offset_in_file,
                obj.clip_offset_in_file + 5,
            ),
        )


class ResultSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    distance = serializers.FloatField()
    entity = EntitySerializer()


class AudioFileSerializer(serializers.Serializer):
    audio_file = serializers.FileField()


class EmbedResultSerializer(serializers.Serializer):
    embedding = serializers.ListField()


class UploadResultSerializer(serializers.Serializer):
    bucket_name = serializers.CharField()
    filename = serializers.CharField()


class CapabilitiesSerializer(serializers.Serializer):
    public_storage = serializers.BooleanField()


class SearchSerializer(serializers.Serializer):
    audio_file = serializers.FileField(required=False, allow_null=True)
    embed = serializers.CharField(required=False, allow_null=True)
    metric_type = serializers.CharField(required=False, allow_null=True)
    nprobe = serializers.IntegerField(required=False, allow_null=True)
    limit = serializers.IntegerField(
        min_value=1, max_value=16384, required=False, allow_null=True
    )
    offset = serializers.IntegerField(required=False, allow_null=True)
    expression = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )

    def validate(self, data):
        if data["embed"] or data["audio_file"]:
            return data
        else:
            raise serializers.ValidationError(
                "embed or audio_file fields can not be both null."
            )


@extend_schema(
    request=SearchSerializer,
    responses={200: ResultSerializer(many=True)},
)
@api_view(["POST"])
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

    if search_params.get("embed") is None:
        status_code, embed_data = get_embedding(search_params.get("audio_file"))
        embed = embed_data["embedding"]

        if status_code != 200:
            return Response(embed_data, 400)
    else:
        embed = json.loads(search_params.get("embed"))

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
    request=AudioFileSerializer(),
    responses={200: UploadResultSerializer},
)
@api_view(["POST"])
@permission_classes([AllowAny])
@parser_classes([FormParser, MultiPartParser])
def upload_audio_view(request):
    """Upload audio file and returns the bucket name and the uploaded filename."""
    if not has_public_storage_enabled():
        return Response(
            {"error": "Public file storage is not enabled in the server."}, 400
        )

    serializer = AudioFileSerializer(
        data={
            "audio_file": request.FILES.get("audio_file"),
        }
    )

    if not serializer.is_valid():
        return Response(serializer.errors, 400)

    validated_data = {**serializer.validated_data}
    name, extension = validated_data.get("audio_file").name.split(".")
    filename = "{}-{}-{}.{}".format(
        datetime.today().date().isoformat(),
        gen_uuid(),
        clean_string(name)[:10],
        extension,
    )
    try:
        minio_result = upload_file(validated_data.get("audio_file"), filename)
        result = UploadResultSerializer(
            {
                "bucket_name": minio_result.bucket_name,
                "filename": minio_result.object_name,
            }
        )

        return Response(result.data, 200)
    except S3Error as exc:
        return Response({"error": exc}, 500)


@extend_schema(
    request=AudioFileSerializer(),
    responses={200: EmbedResultSerializer},
)
@api_view(["POST"])
@permission_classes([AllowAny])
@parser_classes([FormParser, MultiPartParser])
def embed_view(request):
    """Get the embedding array for an audio file. Audio length must be 5 seconds."""
    serializer = AudioFileSerializer(
        data={
            "audio_file": request.FILES.get("audio_file"),
        }
    )

    if not serializer.is_valid():
        return Response(serializer.errors, 400)

    validated_data = {**serializer.validated_data}
    status_code, embed_data = get_embedding(validated_data.get("audio_file"))

    return Response(embed_data, status_code)


@extend_schema(responses={200: CapabilitiesSerializer})
@api_view(["GET"])
@permission_classes([AllowAny])
def capabilities(request):
    public_storage = has_public_storage_enabled()
    serializer = CapabilitiesSerializer({"public_storage": public_storage})

    return Response(serializer.data)


def get_embedding(audio_file):
    """Make a request to the Embed Service URL and return the JSON response."""
    files = {"audio_file": audio_file}
    req = requests.post(
        settings.EMBED_SERVICE_URL,
        files=files,
    )
    return [req.status_code, req.json()]


def pca_transform(embedding):
    """Tranform the search vector from 1280 to 256 dimension numpy array."""
    embed_np = array(embedding).astype("float32")
    reduced_search_vectors = pca_matrix.apply(embed_np[0])
    return reduced_search_vectors


def gen_uuid():
    "Generates a 5 digit random string."
    return "".join(
        secrets.choice(string.ascii_lowercase + string.digits) for i in range(5)
    )


def clean_string(name):
    "Restrict the string to have only letters, numbers and _-+. characters"
    allowed_chars = string.ascii_letters + string.digits + "_-+."
    return "".join(i for i in name if i in allowed_chars)


def has_public_storage_enabled():
    "Check if MINIO is configured."
    public_storage = (
        settings.MINIO.get("SERVER_URL")
        and settings.MINIO.get("SECRET_KEY")
        and settings.MINIO.get("ACCESS_KEY")
    )
    return True if public_storage else False
