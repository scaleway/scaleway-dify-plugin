import logging
from collections.abc import Generator

from dify_plugin.entities.model import AIModelEntity, ModelFeature
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel

from models.common import prepare_credentials

logger = logging.getLogger(__name__)

# Unsupported request fields.
UNSUPPORTED_PARAMETERS = "frequency_penalty"

TOOL_FEATURES = (ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL)


class ScalewayLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    OpenAI-compatible chat models served by Scaleway.
    """

    def _credentials(self, model: str, credentials: dict) -> dict:
        prepared = prepare_credentials(credentials)
        prepared["mode"] = "chat"

        if "function_calling_type" not in prepared:
            schema = self.get_model_schema(model)
            features = (schema.features or []) if schema else []
            if any(feature in features for feature in TOOL_FEATURES):
                prepared["function_calling_type"] = "tool_call"

        return prepared

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: list[PromptMessageTool] | None = None,
        stop: list[str] | None = None,
        stream: bool = True,
        user: str | None = None,
    ) -> LLMResult | Generator:
        parameters = {k: v for k, v in model_parameters.items() if k not in UNSUPPORTED_PARAMETERS}
        if stream:
            # Ask for token usage in the last streamed chunk, so Dify logs real counts.
            parameters["stream_options"] = {"include_usage": True}
        return super()._invoke(
            model,
            self._credentials(model, credentials),
            prompt_messages,
            parameters,
            tools,
            stop,
            stream,
            None,  # User field is not supported
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None = None,
    ) -> int:
        return super().get_num_tokens(model, self._credentials(model, credentials), prompt_messages, tools)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        super().validate_credentials(model, self._credentials(model, credentials))

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, self._credentials(model, credentials))
        entity.parameter_rules = [rule for rule in entity.parameter_rules if rule.name not in UNSUPPORTED_PARAMETERS]
        return entity
