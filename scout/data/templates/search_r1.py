# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Search-R1 template.

This is the default template used in scout/data/preprocess.py with detailed
user instructions for reasoning and search tool usage.
"""

from . import PromptTemplate, register_template

TEMPLATE_NAME = "search_r1"

DESCRIPTION = "Default Search-R1 template with detailed reasoning and search instructions"

SYSTEM_CONTENT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

USER_CONTENT_PREFIX = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <tool_call>{"name": "search", "arguments": {"query_list": ["query"]}}</tool_call> and it will return the top searched results between <tool_response> and </tool_response>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


# Register this template
template = PromptTemplate(
    name=TEMPLATE_NAME, description=DESCRIPTION, system_content=SYSTEM_CONTENT, user_content_prefix=USER_CONTENT_PREFIX
)
register_template(template)
