# bioacoustics-api

[![Built with](https://img.shields.io/badge/Built_with-Cookiecutter_Django_Rest-F7B633.svg)](https://github.com/agconti/cookiecutter-django-rest)

Google Bioacoustics API used by the [Australian Acoustic Observatory Search](https://search.acousticobservatory.org/) project. Check out the project's [documentation](https://devseed.com/api-docs/?url=https://api.search.acousticobservatory.org/api/v1/openapi).

# Prerequisites

- [Docker](https://docs.docker.com/docker-for-mac/install/)  

# Local Development

Start the dev server for local development:
```bash
docker-compose up
```

Run a command inside the docker container:

```bash
docker-compose run --rm web [command]
```
