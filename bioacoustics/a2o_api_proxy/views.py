from urllib.parse import urljoin

from django.conf import settings
from django.views.decorators.cache import cache_page
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes

import requests


@api_view(['GET'])
@permission_classes([AllowAny])
@cache_page(60*60*4)  # 4 hours of cache
def sites_list(request):
    """Get the list of sites from the Acoustic Observatory data API."""
    query_params = request.query_params.urlencode()
    status, data = make_request('sites/?{}'.format(query_params))
    return Response(data, status)


@api_view(['GET'])
@permission_classes([AllowAny])
@cache_page(60*60*4)  # 4 hours of cache
def sites_detail(request, id):
    """Get the detail about a site from the Acoustic Observatory data API."""
    status, data = make_request('sites/{}'.format(id))
    return Response(data, status)


@api_view(['GET'])
@permission_classes([AllowAny])
@cache_page(60*60)  # 1 hour of cache
def recordings_detail(request, id):
    """Get information about an Audio Recording from the Acoustic Observatory data API."""
    status, data = make_request('audio_recordings/{}'.format(id))
    return Response(data, status)


def make_request(endpoint):
    """Make a request to the Embed Service URL and return the JSON response."""
    url = urljoin(settings.A2O_API_URL, endpoint)
    headers = {'Content-Type': 'application/json'}
    if (settings.A2O_API_TOKEN):
        headers['Authorization'] = 'Token {}'.format(settings.A2O_API_TOKEN)

    # verify=False is necessary to bypass SSL Error
    req = requests.get(url, headers=headers, verify=False)
    return [req.status_code, req.json()]


@api_view(['GET'])
@permission_classes([AllowAny])
def recordings_download(request, id, extension):
    """
    Download audio recordings from Acoustic Observatory data API in multiple file formats.
    """
    query_params = request.query_params.urlencode()
    url = urljoin(
        settings.A2O_API_URL,
        "audio_recordings/{}/media.{}?{}".format(id, extension, query_params)
    )
    print(url)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        print('success')
        response = HttpResponse(
            response.iter_content(chunk_size=1024),
            content_type=response.headers['Content-Type']
        )
        response['Content-Disposition'] = 'attachment; filename="{}.{}"'.format(id, extension)
        return response

    return HttpResponse("Error: Failed to download audio file", status=response.status_code)
