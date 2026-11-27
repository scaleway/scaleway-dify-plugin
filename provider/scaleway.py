from collections.abc import Mapping

import requests
from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

from models.common import prepare_credentials


class ScalewayModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """Check the API key by listing models from /v1/models"""
        prepared = prepare_credentials(dict(credentials))
        try:
            response = requests.get(
                f"{prepared['endpoint_url']}/models",
                headers={"Authorization": f"Bearer {prepared.get('api_key', '')}"},
                timeout=(10, 30),
            )
        except requests.RequestException as ex:
            raise CredentialsValidateFailedError(f"Could not reach {prepared['endpoint_url']}: {ex}") from ex

        if response.status_code != 200:
            raise CredentialsValidateFailedError(
                f"Scaleway rejected the credentials (HTTP {response.status_code}): {response.text[:300]}"
            )
