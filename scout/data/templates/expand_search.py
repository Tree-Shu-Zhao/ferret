"""
ExpandSearch template.

This template extends the Search-R1 approach by generating multiple diverse query
variants (paraphrases, decomposed sub-questions, keyword expansions) to facilitate
retrieval for more relevant knowledge. Each search generates 3 diverse queries that
are processed in parallel.
"""

from . import PromptTemplate, register_template

TEMPLATE_NAME = "expand_search"

DESCRIPTION = "ExpandSearch template with query expansion support for generating diverse search query variants"

SYSTEM_CONTENT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

USER_CONTENT_PREFIX = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call>{"name": "search", "arguments": {"query_list": ["query"]}}</tool_call> and it will return the top searched results between <tool_response> and </tool_response>. \
Within the query_list, generate 3 diverse query variants (must be included in a SINGLE tool calling) — such as paraphrases, decomposed sub-questions, keyword expansions — to facilitate retrieval for more relevant knowledge. Never split related queries across multiple consecutive tool_calls; instead, combine all query variants into one query_list array.\
These queries will be run in parallel to retrieve comprehensive information. \
For example: <tool_call>{"name": "search", "arguments": {"query_list": ["query variant 1", "query variant 2", "query variant 3"]}}</tool_call>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


# Register this template
template = PromptTemplate(
    name=TEMPLATE_NAME, description=DESCRIPTION, system_content=SYSTEM_CONTENT, user_content_prefix=USER_CONTENT_PREFIX
)
register_template(template)
