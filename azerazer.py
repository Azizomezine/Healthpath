from aiml_openai.embeddings.base import AIMLEmbeddings

from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import BoolInput, DictInput, DropdownInput, FloatInput, IntInput, MessageTextInput, SecretStrInput


class AIMLEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "AIML Embeddings"
    description = "Generate embeddings using AIML models."
    icon = "AIML"
    name = "AIMLEmbeddings"

    inputs = [
        DictInput(
            name="default_headers",
            display_name="Default Headers",
            advanced=True,
            info="Default headers to use for the API request.",
        ),
        DictInput(
            name="default_query",
            display_name="Default Query",
            advanced=True,
            info="Default query parameters to use for the API request.",
        ),
        IntInput(name="chunk_size", display_name="Chunk Size", advanced=True, value=1000),
        MessageTextInput(name="client", display_name="Client", advanced=True),
        MessageTextInput(name="deployment", display_name="Deployment", advanced=True),
        IntInput(name="embedding_ctx_length", display_name="Embedding Context Length", advanced=True, value=1536),
        IntInput(name="max_retries", display_name="Max Retries", value=3, advanced=True),
        DropdownInput(
            name="model",
            display_name="Model",
            advanced=False,
            options=[
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            value="text-embedding-3-small",
        ),
        DictInput(name="model_kwargs", display_name="Model Kwargs", advanced=True),
        SecretStrInput(name="aiml_api_base", display_name="AIML API Base", advanced=True, value="https://api.aimlapi.com/v1"),
        SecretStrInput(name="aiml_api_key", display_name="AIML API Key", value="AIMLAPI_API_KEY"),
        SecretStrInput(name="aiml_api_type", display_name="AIML API Type", advanced=True),
        MessageTextInput(name="aiml_api_version", display_name="AIML API Version", advanced=True),
        MessageTextInput(
            name="aiml_organization",
            display_name="AIML Organization",
            advanced=True,
        ),
        MessageTextInput(name="aiml_proxy", display_name="AIML Proxy", advanced=True),
        FloatInput(name="request_timeout", display_name="Request Timeout", advanced=True),
        BoolInput(name="show_progress_bar", display_name="Show Progress Bar", advanced=True),
        BoolInput(name="skip_empty", display_name="Skip Empty", advanced=True),
        MessageTextInput(
            name="tiktoken_model_name",
            display_name="TikToken Model Name",
            advanced=True,
        ),
        BoolInput(
            name="tiktoken_enable",
            display_name="TikToken Enable",
            advanced=True,
            value=True,
            info="If False, you must have transformers installed.",
        ),
        IntInput(
            name="dimensions",
            display_name="Dimensions",
            info="The number of dimensions the resulting output embeddings should have. Only supported by certain models.",
            advanced=True,
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        return AIMLEmbeddings(
            tiktoken_enabled=self.tiktoken_enable,
            default_headers=self.default_headers,
            default_query=self.default_query,
            allowed_special="all",
            disallowed_special="all",
            chunk_size=self.chunk_size,
            deployment=self.deployment,
            embedding_ctx_length=self.embedding_ctx_length,
            max_retries=self.max_retries,
            model=self.model,
            model_kwargs=self.model_kwargs,
            base_url=self.aiml_api_base,
            api_key=self.aiml_api_key,
            api_type=self.aiml_api_type,
            api_version=self.aiml_api_version,
            organization=self.aiml_organization,
            proxy=self.aiml_proxy,
            timeout=self.request_timeout or None,
            show_progress_bar=self.show_progress_bar,
            skip_empty=self.skip_empty,
            tiktoken_model_name=self.tiktoken_model_name,
            dimensions=self.dimensions or None,
        )
