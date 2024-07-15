import operator
from functools import reduce

from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from langflow.base.constants import STREAM_INFO_TEXT
from langflow.base.models.model import LCModelComponent
from langflow.base.models.openai_constants import MODEL_NAMES
from langflow.field_typing import LanguageModel
from langflow.inputs import (
    BoolInput,
    DictInput,
    DropdownInput,
    FloatInput,
    IntInput,
    MessageInput,
    SecretStrInput,
    StrInput,
)

class AIMLAPIModelComponent(LCModelComponent):
    display_name = "AIMLAPI"
    description = "Generates text using AIMLAPI LLMs."
    icon = "OpenAI"
    name = "AIMLAPIModel"

    inputs = [
        MessageInput(name="input_value", display_name="Input"),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
        ),
        DictInput(name="model_kwargs", display_name="Model Kwargs", advanced=True),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, it will output JSON regardless of passing a schema.",
        ),
        DictInput(
            name="output_schema",
            is_list=True,
            display_name="Schema",
            advanced=True,
            info="The schema for the Output of the model. You must pass the word JSON in the prompt. If left blank, JSON mode will be disabled.",
        ),
        DropdownInput(
            name="model_name", display_name="Model Name", advanced=False, options=MODEL_NAMES, value=MODEL_NAMES[0]
        ),
        StrInput(
            name="aimlapi_base",
            display_name="AIMLAPI Base",
            advanced=True,
            info="The base URL of the AIMLAPI API. Defaults to https://api.aimlapi.com.",
        ),
        SecretStrInput(
            name="aimlapi_api_key",
            display_name="AIMLAPI API Key",
            info="The AIMLAPI API Key to use for the AIMLAPI model.",
            advanced=False,
            value="AIMLAPI_API_KEY",
        ),
        FloatInput(name="temperature", display_name="Temperature", value=0.1),
        BoolInput(name="stream", display_name="Stream", info=STREAM_INFO_TEXT, advanced=True),
        StrInput(
            name="system_message",
            display_name="System Message",
            info="System message to pass to the model.",
            advanced=True,
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        output_schema_dict: dict[str, str] = reduce(operator.ior, self.output_schema or {}, {})
        aimlapi_api_key = self.aimlapi_api_key
        temperature = self.temperature
        model_name: str = self.model_name
        max_tokens = self.max_tokens
        model_kwargs = self.model_kwargs or {}
        aimlapi_base = self.aimlapi_base or "https://api.aimlapi.com"
        json_mode = bool(output_schema_dict) or self.json_mode
        seed = self.seed
        model_kwargs["seed"] = seed

        if aimlapi_api_key:
            api_key = SecretStr(aimlapi_api_key)
        else:
            api_key = None
        output = ChatOpenAI(
            max_tokens=max_tokens or None,
            model_kwargs=model_kwargs,
            model=model_name,
            base_url=aimlapi_base,
            api_key=api_key,
            temperature=temperature or 0.1,
        )
        if json_mode:
            if output_schema_dict:
                output = output.with_structured_output(schema=output_schema_dict, method="json_mode")  # type: ignore
            else:
                output = output.bind(response_format={"type": "json_object"})  # type: ignore

        return output  # type: ignore

    def _get_exception_message(self, e: Exception):
        try:
            from openai import BadRequestError
        except ImportError:
            return
        if isinstance(e, BadRequestError):
            message = e.body.get("message")  # type: ignore
            if message:
                return message
        return