from minio import Minio

from django.conf import settings


def upload_file(audio_file, filename):
    """Upload file to the MINIO server"""
    MINIO_CONFIG = settings.MINIO
    is_localhost = (
        MINIO_CONFIG["SERVER_URL"].startswith("127.0.0.1")
        or MINIO_CONFIG["SERVER_URL"].startswith("192.168.")
        or MINIO_CONFIG["SERVER_URL"].startswith("localhost")
    )
    client = Minio(
        MINIO_CONFIG["SERVER_URL"],
        access_key=MINIO_CONFIG["ACCESS_KEY"],
        secret_key=MINIO_CONFIG["SECRET_KEY"],
        secure=False if is_localhost else True,
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
