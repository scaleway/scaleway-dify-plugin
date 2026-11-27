from typing import IO

from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
)
from dify_plugin.interfaces.model.openai_compatible.speech2text import (
    OAICompatSpeech2TextModel,
)

from models.common import prepare_credentials

FILE_UPLOAD_LIMIT_MB = 25
SUPPORTED_FILE_EXTENSIONS = "flac,m4a,mp2,mp3,mp4,mpeg,ogg,wav,webm"


class ScalewaySpeech2TextModel(OAICompatSpeech2TextModel):
    """
    OpenAI-compatible speech2text models served by Scaleway.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        file: IO[bytes],
        user: str | None = None,
    ) -> str:
        return super()._invoke(model, prepare_credentials(credentials), file, None)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        super().validate_credentials(model, prepare_credentials(credentials))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        # The SDK has no schema for speech-to-text models added by ID.
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.SPEECH2TEXT,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.FILE_UPLOAD_LIMIT: FILE_UPLOAD_LIMIT_MB,
                ModelPropertyKey.SUPPORTED_FILE_EXTENSIONS: SUPPORTED_FILE_EXTENSIONS,
            },
            parameter_rules=[],
        )
