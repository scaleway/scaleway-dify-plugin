from dify_plugin.entities.model import AIModelEntity, EmbeddingInputType, ModelPropertyKey
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult
from dify_plugin.interfaces.model.openai_compatible.text_embedding import (
    OAICompatEmbeddingModel,
)

from models.common import prepare_credentials

DEFAULT_MAX_CHUNKS = 32


class ScalewayTextEmbeddingModel(OAICompatEmbeddingModel):
    """
    OpenAI-compatible embedding models served by Scaleway.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: str | None = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        return super()._invoke(
            model,
            prepare_credentials(credentials),
            texts,
            None,  # Scaleway does not support the "user" field.
            input_type,
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        super().validate_credentials(model, prepare_credentials(credentials))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)
        # The SDK sends one text per request for models added by ID; use the
        # batch size from the form instead.
        entity.model_properties[ModelPropertyKey.MAX_CHUNKS] = int(credentials.get("max_chunks") or DEFAULT_MAX_CHUNKS)
        return entity
