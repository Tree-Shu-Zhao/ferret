"""
ParallelSearch reward function for evaluating agent responses.

This reward function extends the Search-R1 approach by:
1. Rewarding correct decomposition of complex queries into sub-questions
2. Adjusting search count expectations based on question type
3. Providing additional metrics for tracking decomposition behavior

Paper notation:
- λ_d (lambda_d): Decomposition reward weight
- λ_s (lambda_s): Search count reward weight
- α (alpha): Reward multiplier for correct decomposition (default: 3.0)
"""

import re
import random
import json
from ferret.reward_score.search_r1_format import (
    em_check,
    is_valid_sequence,
    extract_solution,
)


def extract_search_decomposition_info(solution_str):
    """Extract search query information and detect decomposition.

    Expects OpenAI function format: {"name": "search", "arguments": {"query_list": [...]}}

    Returns:
        tuple: (total_decomposed_queries, has_decomposition)
            - total_decomposed_queries: Total number of individual queries across all tool_calls
            - has_decomposition: True if any tool_call contains multiple queries in query_list
    """
    # Pattern to extract tool_call content
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, solution_str, re.DOTALL)

    total_queries = 0
    has_decomposition = False

    for match in matches:
        try:
            # Clean up the match content and parse as JSON
            content = match.strip()
            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                continue

            # Extract query_list from OpenAI function format
            if 'arguments' in parsed and isinstance(parsed['arguments'], dict):
                query_list = parsed['arguments'].get('query_list')
                if query_list and isinstance(query_list, list):
                    query_count = len(query_list)
                    total_queries += query_count
                    # Decomposition detected if any tool_call has multiple queries
                    if query_count > 1:
                        has_decomposition = True

        except (json.JSONDecodeError, ValueError):
            # If not valid JSON, ignore this tool_call
            pass

    return total_queries, has_decomposition


def compute_score_em(
    solution_str, ground_truth, data_source, extra_info,
    structure_format_score=0.1, final_format_score=0.0,
    retrieval_score=0.1, score=1.0,
    lambda_d=0.15, lambda_s=0.35, alpha=3.0,
    *args, **kwargs):
    """The scoring function for ParallelSearch with decomposition rewards.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth dictionary with 'target' key
        data_source: the data source identifier
        extra_info: extra information containing question type
        structure_format_score: score for valid structure format (default: 0.1)
        final_format_score: score for partial format (default: 0.0)
        retrieval_score: score for correct retrieval (default: 0.1)
        score: the score for the correct answer (default: 1.0)
        lambda_d: decomposition reward weight (λ_d in paper) (default: 0.15)
        lambda_s: search count reward weight (λ_s in paper) (default: 0.35)
        alpha: reward multiplier for correct decomposition (α in paper) (default: 3.0)

    Returns:
        dict with 'score' key and additional metrics for tracking
    """
    # Extract question type from extra_info
    question_type = extra_info.get("type", "") if extra_info else ""

    # Map question type to whether decomposition is required
    # comparison questions (requires_decomposition=1) should use query decomposition
    # na questions (requires_decomposition=-1) are not applicable
    # regular questions (requires_decomposition=0) don't need decomposition
    if question_type == "comparison":
        requires_decomposition = 1
    elif question_type == "na":
        requires_decomposition = -1
    else:
        requires_decomposition = 0

    # Check format validity
    is_valid_format, _ = is_valid_sequence(solution_str)

    # Check retrieval correctness (currently disabled as in original)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = False  # Keeping original logic: is_retrieval_correct disabled

    # Extract answer
    answer = extract_solution(solution_str=solution_str)

    # Extract search queries and detect decomposition
    total_queries, has_decomposition = extract_search_decomposition_info(solution_str)

    # Calculate decomposition reward (λ_d term in paper)
    decomposition_reward = 0
    if has_decomposition and requires_decomposition > 0:
        # Correctly decomposed when needed (comparison questions): λ_d * α
        decomposition_reward += lambda_d * alpha
    elif not has_decomposition and requires_decomposition <= 0:
        # Correctly didn't decompose when not needed: λ_d
        decomposition_reward += lambda_d

    # Debug printing (randomly sample 1/64 for logging)
    do_print = random.randint(1, 64) == 1

    # Count number of search attempts
    num_tool_calls = solution_str.count("<tool_call>")

    # Calculate search count reward (λ_s term in paper) based on question type
    if requires_decomposition != 0:
        # For comparison/na questions, expect around 1 tool call
        search_count_reward = lambda_s - abs(num_tool_calls - 1) * lambda_s
    else:
        # For regular questions, expect up to 2 tool calls
        search_count_reward = lambda_s - abs(min(num_tool_calls, 2) - 2) * lambda_s

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth.get('target', 'N/A')}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"Question type: {question_type}")
        print(f"has_decomposition: {has_decomposition}")
        print(f"total_queries: {total_queries}")

    # Check answer correctness
    answer_correct = False
    if answer is not None and 'target' in ground_truth:
        answer_correct = em_check(answer, ground_truth['target'])

    # Calculate final reward based on answer correctness and format
    if answer is None:
        # No answer extracted
        if is_valid_format:
            final_reward = structure_format_score + decomposition_reward + search_count_reward
        else:
            final_reward = decomposition_reward + search_count_reward
    else:
        # Answer was extracted
        if answer_correct:
            # Correct answer
            if is_valid_format:
                final_reward = score + decomposition_reward + search_count_reward
            else:
                final_reward = score - structure_format_score + decomposition_reward + search_count_reward
        elif is_valid_format:
            # Wrong answer but valid format
            final_reward = structure_format_score + decomposition_reward + search_count_reward
        else:
            # Wrong answer and invalid format
            final_reward = final_format_score + decomposition_reward + search_count_reward

    # Return reward with detailed metrics
    # The NaiveRewardManager expects a dict with "score" key
    return {
        "score": final_reward,  # Required: the actual reward value
        # Additional metrics (automatically collected as reward_extra_info)
        "format_valid": 1.0 if is_valid_format else 0.0,
        "has_answer": 1.0 if answer is not None else 0.0,
        "acc": 1.0 if answer_correct else 0.0,
        "is_comparison_question": 1.0 if requires_decomposition > 0 else 0.0,
        "has_decomposition": 1.0 if has_decomposition else 0.0,
        "decomposition_correct": 1.0 if (has_decomposition and requires_decomposition > 0) or (not has_decomposition and requires_decomposition <= 0) else 0.0,
        "total_queries": float(total_queries),
        "num_tool_calls": float(num_tool_calls),
        "decomposition_reward": decomposition_reward,
        "search_count_reward": search_count_reward,
        "reward_final": final_reward,
    }
