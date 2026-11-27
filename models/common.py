"""Settings shared by all Scaleway model types."""

SERVERLESS_URL = "https://api.scaleway.ai"
DEFAULT_ENDPOINT_URL = f"{SERVERLESS_URL}/v1"


def build_endpoint_url(credentials: dict) -> str:
    """Pick the API base URL from the credentials.

    1. An explicit API Base URL (used for dedicated deployments).
    2. Otherwise a Project ID gives the project-scoped serverless URL.
    3. Otherwise the default serverless URL, which uses the default project.
    """
    endpoint_url = (credentials.get("endpoint_url") or "").strip()
    if endpoint_url:
        return endpoint_url.rstrip("/")
    project_id = (credentials.get("project_id") or "").strip().strip("/")
    if project_id:
        return f"{SERVERLESS_URL}/{project_id}/v1"
    return DEFAULT_ENDPOINT_URL


def prepare_credentials(credentials: dict) -> dict:
    """Return a copy of the credentials with the resolved API base URL."""
    prepared = dict(credentials)
    prepared["endpoint_url"] = build_endpoint_url(prepared)
    return prepared
