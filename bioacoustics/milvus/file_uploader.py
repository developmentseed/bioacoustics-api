from minio import Minio

from django.conf import settings


def upload_file(audio_file, filename):
    """Upload file to the MINIO server"""
    MINIO_CONFIG = settings.MINIO
    client = Minio(
        MINIO_CONFIG["SERVER_URL"],
        access_key=MINIO_CONFIG["ACCESS_KEY"],
        secret_key=MINIO_CONFIG["SECRET_KEY"],
        region=MINIO_CONFIG["BUCKET_LOCATION"],
        secure=MINIO_CONFIG["SSL_SUPPORTED"],
    )

    # Create bucket if it doesn't exist
    found = client.bucket_exists(MINIO_CONFIG["BUCKET_NAME"])
    if not found:
        client.make_bucket(MINIO_CONFIG["BUCKET_NAME"])

    result = client.put_object(
        MINIO_CONFIG["BUCKET_NAME"],
        filename,
        audio_file,
        audio_file.size,
        content_type=audio_file.content_type,
    )
    return result
