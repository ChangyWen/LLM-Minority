import json
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, HttpOptions, ThinkingConfig, GroundingMetadata
from dotenv import load_dotenv
load_dotenv()


class Gemini:
    def __init__(self):
        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.google_search_tool = Tool(google_search = GoogleSearch())
        print("gemini initialized")

    def _format_search_results(self, grounding_metadata: GroundingMetadata):
        result = {}
        result["web_search_queries"] = []
        result["grounding_chunks"] = []
        result["groundingSupports"] = []
        for web_search_query in grounding_metadata.web_search_queries:
            result["web_search_queries"].append(web_search_query)
        for grounding_chunk in grounding_metadata.grounding_chunks:
            tmp = {}
            try:
                tmp["web"] = {
                    "domain": grounding_chunk.web.domain,
                    "title": grounding_chunk.web.title,
                    "uri": grounding_chunk.web.uri
                }
            except:
                print(f"Error in getting web: {grounding_chunk}")
            try:
                if grounding_chunk.retrieved_context:
                    tmp["retrieved_context"] = {
                        "title": grounding_chunk.retrieved_context.title,
                        "uri": grounding_chunk.retrieved_context.uri,
                        "text": grounding_chunk.retrieved_context.text
                    }
            except:
                print(f"Error in getting retrieved_context: {grounding_chunk}")
            result["grounding_chunks"].append(tmp)
        for grounding_support in grounding_metadata.grounding_supports:
            tmp = {}
            tmp["confidence_scores"] = grounding_support.confidence_scores
            tmp["grounding_chunk_indices"] = grounding_support.grounding_chunk_indices
            tmp["segment_text"] = grounding_support.segment.text
            tmp["segment_start_index"] = grounding_support.segment.start_index
            tmp["segment_end_index"] = grounding_support.segment.end_index
            tmp["segment_part_index"] = grounding_support.segment.part_index
            result["groundingSupports"].append(tmp)
        return result

    def chat(self, prompt, model_name, enable_search, enable_thinking, thinking_budget, temperature=1.0, top_p=1.0):
        try:
            if model_name == "gemini-2.5-pro":
                assert enable_thinking == True
            # generate the response
            response = None
            tools = [self.google_search_tool] if enable_search else None
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    thinking_config=ThinkingConfig(
                        thinking_budget=thinking_budget if enable_thinking else 0,
                        include_thoughts=True if enable_thinking else False,
                    ),
                    tools=tools,
                    temperature=temperature,
                    top_p=top_p
                )
            )
            response_thinking_draft = None
            response_text = None
            response_search_results = None
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    response_thinking_draft = part.text
                else:
                    response_text = part.text
            final_response = { "value": response_text }
            if enable_search:
                response_search_results = self._format_search_results(response.candidates[0].grounding_metadata)
                final_response["srouces"] = response_search_results
            if enable_thinking:
                final_response["thinking_draft"] = response_thinking_draft
            return final_response
        except Exception as e:
            print(f"Error in Gemini.chat: {e}")
            return None


# if __name__ == "__main__":
#     gemini = Gemini()
#     print(json.dumps(gemini.chat("How many people live in the US?", "gemini-2.5-flash", False, False, None), indent=4))
#     print(json.dumps(gemini.chat("How many people live in the US?", "gemini-2.5-flash", True, False, None), indent=4))
#     print(json.dumps(gemini.chat("How many people live in the US?", "gemini-2.5-flash", True, True, None), indent=4))
#     print(json.dumps(gemini.chat("How many people live in the US?", "gemini-2.5-pro", False, True, 2000), indent=4))
#     print(json.dumps(gemini.chat("How many people live in the US?", "gemini-2.5-pro", True, True, 2000), indent=4))