from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status


class TestA2OProxyViews(APITestCase):
    def test_sites_list(self):
        url = reverse('sites-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data.get('data')), 25)
        # pass some query params
        response = self.client.get(url, {'items': 5, 'page': 2, 'direction': 'asc'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data.get('data')), 5)

    def test_sites_detail(self):
        url = reverse('sites-detail', kwargs={'id': 213})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        url = reverse('sites-detail', kwargs={'id': 21309090900})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_recordings_detail(self):
        url = reverse('recordings-detail', kwargs={'id': 1090736})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
