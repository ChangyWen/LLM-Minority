from typing import Callable
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from openai import AzureOpenAI
from azure.core.credentials import TokenCredential
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from azure.core.pipeline import PipelineRequest, PipelineContext
from azure.core.rest import HttpRequest

class MSRA:

    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2025-04-01-preview",
            azure_endpoint="https://csnf-singularity-aoai-eastus2.openai.azure.com/",
            azure_ad_token_provider=self.token_provider,
        )
        print("msra initialized")

    def _make_request(self) -> PipelineRequest[HttpRequest]:
        return PipelineRequest(
            HttpRequest("CredentialWrapper", "https://fakeurl"), PipelineContext(None)
        )

    def get_bearer_token_provider(
        credential: TokenCredential, *scopes: str
    ) -> Callable[[], str]:
        policy = BearerTokenCredentialPolicy(credential, *scopes)

        def wrapper(self) -> str:
            request = self._make_request()
            policy.on_request(request)
            return request.http_request.headers["Authorization"][len("Bearer ") :]

        return wrapper

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(
            managed_identity_client_id="e6162a0d-e540-4454-995f-30bcb97f35b4"
        ),
        "https://cognitiveservices.azure.com/.default",
    )

    def chat(self, prompt, model_name, enable_search, enable_thinking, reasoning_effort, temperature=1.0, top_p=1.0):
        try:
            # remove the "msra-" prefix
            model_name = model_name.replace("msra-", "")
            assert model_name == "gpt-4.1-nano" or model_name == "o4-mini" or model_name == "gpt-5" or model_name == "gpt-4o"
            if model_name == "gpt-4o":
                model_name = "csnf-gpt-4o"
            assert not enable_search

            if model_name == "gpt-4.1-nano" or model_name == "csnf-gpt-4o":
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{ "role": "user", "content": prompt }],
                    temperature=temperature,
                    top_p=top_p,
                )
                return { "value": response.choices[0].message.content }
            elif model_name == "o4-mini" or model_name == "gpt-5":
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{ "role": "user", "content": prompt }],
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    top_p=top_p,
                )
                return { "value": response.choices[0].message.content }
        except Exception as e:
            print(f"Error in MSRA.chat: {e}")
            return None