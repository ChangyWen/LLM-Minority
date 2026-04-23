import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageRole
from dotenv import load_dotenv
load_dotenv()


#### gpt ####
class GPT:
    def __init__(self):
        self.client = AzureOpenAI(azure_endpoint=os.getenv("OPENAI_ENDPOINT"), api_key=os.getenv("OPENAI_API_KEY"), api_version="2025-04-01-preview")
        # self.bing_search_project_client, self.bing_search_agent = self._get_bing_search_agent()
        print("gpt initialized")

    def _format_search_results(self, content):
        for item in content:
            if item is None: continue
            if hasattr(item, 'text') and hasattr(item.text, 'value'):
                return {
                    'value': item.text.value,
                    'srouces': [
                        {
                            'type': a.type,
                            'text': a.text,
                            'start_index': a.start_index,
                            'end_index': a.end_index,
                            'url_citation': {
                                'url': a.url_citation.url if a.url_citation else None,
                                'title': a.url_citation.title if a.url_citation else None
                            }
                        }
                        for a in item.text.annotations if a is not None
                    ]
                }
        return None

    ##### bing search #####
    def _get_bing_search_agent(self):
        # Create an Azure AI Client from an endpoint, copied from your Azure AI Foundry project.
        # You need to login to Azure subscription via Azure CLI and set the environment variables
        project_endpoint = os.getenv("AZURE_BING_ENDPOINT")  # Ensure the PROJECT_ENDPOINT environment variable is set
        # Create an AIProjectClient instance
        project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential(),  # Use Azure Default Credential for authentication
            # api_version="2025-02-01-Preview", # use default version
        )
        agent = project_client.agents.get_agent(agent_id=os.getenv("AZURE_BING_AGENT_ID"))
        # print(f"Fetched agent, ID: {agent.id}")
        return (project_client, agent)

    def _gpt_with_bing_search(self, prompt):
        # currently supporting only gpt-4.1 with bing search
        try:
            thread = self.bing_search_project_client.agents.threads.create()
            # Add a message to the thread
            message = self.bing_search_project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",  # Role of the message sender
                content=prompt
            )
            # Create and process an agent run
            run = self.bing_search_project_client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=self.bing_search_agent.id,
                tool_choice={"type": "bing_grounding"}  # force the model to use Grounding with Bing Search tool
            )
            # Check if the run failed
            if run.status == "failed":
                print(f"gpt_with_bing_search failed: {run.last_error}")
                return None
            else:
                # Fetch and log all messages
                messages = self.bing_search_project_client.agents.messages.list(thread_id=thread.id)
                for message in messages:
                    if message.role == MessageRole.AGENT:
                        return self._format_search_results(message.content)
            # delete the thread
            # self.bing_search_project_client.agents.threads.delete(thread_id=thread.id) ## causing rate limit error
        except Exception as e:
            print(f"Error in GPT._gpt_with_bing_search: {e}")
            return None

    def chat(self, prompt, model_name, enable_search, enable_thinking, reasoning_effort, temperature=1.0, top_p=1.0):
        try:
            if enable_search:
                return self._gpt_with_bing_search(prompt)
            else:
                if model_name == "gpt-4.1":
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{ "role": "user", "content": prompt }],
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return { "value": response.choices[0].message.content }
                elif model_name == "o4-mini":
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{ "role": "user", "content": prompt }],
                        reasoning_effort=reasoning_effort,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return { "value": response.choices[0].message.content }
        except Exception as e:
            print(f"Error in GPT.chat: {e}")
            return None


# if __name__ == "__main__":
#     gpt = GPT()
#     print(gpt.chat("How many people live in the US?", "gpt-4.1", True, False, "low"))
#     print(gpt.chat("How many people live in the US?", "gpt-4.1", False, False, "low"))
#     print(gpt.chat("How many people live in the US?", "o4-mini", False, True, "low"))