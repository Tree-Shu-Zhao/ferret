"""
SearchToolWithSqueezer extends the SearchTool with LLM-based compression of search results.
This tool is designed for the ExpandSearch approach where diverse query variants
are used to retrieve comprehensive information, which is then compressed by an LLM.
"""

import os
import json
from typing import List, Any, Dict, Optional
import logging

from verl.tools.search_tool import SearchTool
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger(__name__)


class SearchToolWithSqueezer(SearchTool):
    """
    Extends SearchTool to add LLM-based compression (squeezing) of search results.

    This tool performs search with multiple query variants and then compresses
    the retrieved results using an LLM to provide a concise, relevant answer.

    For ExpandSearch:
    - Receives 3 query variants for one sample
    - Retrieves top-k documents for each query (e.g., 3 queries × 5 docs = 15 docs)
    - Concatenates all retrieved documents
    - Compresses them into one concise answer using LLM
    """

    def __init__(self, config: dict, tool_schema: dict):
        """
        Initialize SearchToolWithSqueezer with additional squeezer configuration.

        Args:
            config: Configuration dictionary containing both SearchTool and squeezer configs
            tool_schema: OpenAI-format tool schema
        """
        super().__init__(config, tool_schema)

        # Squeezer-specific configuration
        self.squeezer_model = config.get('squeezer_model', 'meta/llama-4-maverick-17b-128e-instruct')
        self.squeezer_max_tokens = config.get('squeezer_max_tokens', 512)

        # Create LLM instance for compression
        self._llm = None  # Lazy initialization

    def _create_llm(self):
        """
        Create and return an LLM instance for search result compression.

        Returns:
            ChatNVIDIA instance configured for compression
        """
        if self._llm is None:
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY environment variable is not set")

            self._llm = ChatNVIDIA(
                model=self.squeezer_model,
                max_tokens=self.squeezer_max_tokens,
                temperature=0.0,
                api_key=api_key
            )
        return self._llm

    def _generate_compressed_answer(
        self,
        llm: ChatNVIDIA,
        queries: List[str],
        context: str,
        prompt_template: str
    ) -> str:
        """
        Generate a compressed answer using the LLM.

        Args:
            llm: LLM instance
            queries: List of query strings (all variants for this sample)
            context: Combined context from all retrievals
            prompt_template: Template string with {queries} and {context} placeholders

        Returns:
            Compressed answer string
        """
        # Format queries for display
        formatted_queries = " | ".join(queries)

        # Create prompt
        prompt = prompt_template.format(
            queries=formatted_queries,
            context=context
        )

        try:
            # Generate compressed answer
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error in LLM compression: {e}")
            # Fallback: return truncated context if LLM fails
            return context[:1000] + "..." if len(context) > 1000 else context

    def execute_search(
        self,
        instance_id: str,
        query_list: List[str],
        retrieval_service_url: str,
        topk: int,
        timeout: int
    ) -> tuple[str, dict]:
        """
        Execute search with query list and compress all results using LLM.

        This method overrides the parent's execute_search to add compression
        after retrieval. For ExpandSearch with 3 query variants:
        1. Retrieves top-k documents for each query (e.g., 3×5 = 15 docs total)
        2. Concatenates all retrieved documents
        3. Compresses into one concise answer using LLM

        Args:
            instance_id: Unique instance identifier
            query_list: List of query variants for this single sample
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to retrieve per query
            timeout: Request timeout in seconds

        Returns:
            Tuple of (compressed_result_text, metadata)
        """
        # Step 1: Perform batch search using parent's implementation
        # This returns all search results concatenated for all queries in query_list
        original_result_text, original_metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency
            timeout=timeout
        )

        # If search failed, return original results
        if original_metadata.get('status') != 'success':
            return original_result_text, original_metadata

        try:
            # Step 2: Parse search results
            # The result contains all search results joined together
            search_result_json = json.loads(original_result_text)
            combined_context = search_result_json.get("result", "")

            # Step 3: Create LLM instance for compression
            llm = self._create_llm()

            # Step 4: Define compression prompt template
            prompt_template = """You are a helpful assistant.
You are given a series of queries and some contexts.
Return the answer to the queries based on the contexts and nothing else.
Provide a comprehensive yet concise answer that addresses the queries.

Queries: {queries}
Contexts: {context}
Answer:"""

            # Step 5: Apply LLM compression to the combined context
            # All retrievals for all query variants are compressed into one answer
            compressed_answer = self._generate_compressed_answer(
                llm=llm,
                queries=query_list,
                context=combined_context,
                prompt_template=prompt_template
            )

            # Step 6: Format compressed result
            compressed_result_text = json.dumps(
                {"result": compressed_answer},
                ensure_ascii=False
            )

            # Update metadata
            compressed_metadata = original_metadata.copy()
            compressed_metadata['compressed'] = True
            compressed_metadata['squeezer_model'] = self.squeezer_model
            compressed_metadata['num_queries'] = len(query_list)
            compressed_metadata['docs_per_query'] = topk
            compressed_metadata['total_docs_retrieved'] = len(query_list) * topk

            logger.info(
                f"[SearchToolWithSqueezer] Compressed {len(query_list) * topk} documents "
                f"from {len(query_list)} queries into one answer for instance {instance_id}"
            )

            return compressed_result_text, compressed_metadata

        except Exception as e:
            logger.error(f"Error in result compression: {e}")
            # Fallback to original results if compression fails
            return original_result_text, original_metadata