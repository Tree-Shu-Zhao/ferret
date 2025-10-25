import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Remove role tags (user/assistant) that may appear between structured tags
    content = text

    # First pass: Simply remove lines that contain only role tags
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Only skip lines that are exactly "user" or "assistant"
        if stripped not in ['user', 'assistant']:
            cleaned_lines.append(line)

    # Rejoin the lines
    content = '\n'.join(cleaned_lines)


    # Check for balanced tags - now using think, tool_call, tool_response, answer
    tags_to_check = ["think", "tool_call", "tool_response", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|tool_call|tool_response|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    # start -> [<think> -> <tool_call> -> <tool_response>]* -> <think> -> <answer> -> end
    state = "start"

    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|tool_call|tool_response|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "after_tool_response"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<tool_call>" and state == "after_think":
                state = "in_tool_call"
            elif part == "</tool_call>" and state == "in_tool_call":
                state = "after_tool_call"
            elif part == "<tool_response>" and state == "after_tool_call":
                state = "in_tool_response"
            elif part == "</tool_response>" and state == "in_tool_response":
                state = "after_tool_response"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_tool_call", "in_tool_response", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_tool_call", "after_tool_response"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the answer from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are no matches, return None
    if len(matches) == 0:
        return None

    # Return the last answer tag content
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    """Extract information from <tool_response> tags."""
    pattern = r"<tool_response>(.*?)</tool_response>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str, ground_truth, data_source, extra_info, 
    structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.,
    *args, **kwargs):
    """The scoring function for exact match (EM) with detailed metrics tracking.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        data_source: the data source
        extra_info: extra information
        structure_format_score: score for valid structure format
        final_format_score: score for partial format
        retrieval_score: score for correct retrieval
        format_score: deprecated format score parameter
        score: the score for the correct answer

    Returns:
        dict with 'reward_tensor' and 'reward_extra_info' containing detailed metrics
    """
    is_valid_format, error_msg = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])

    answer = extract_solution(solution_str=solution_str)
    answer_correct = False
    if answer is not None:
        answer_correct = em_check(answer, ground_truth['target'])

    # Count tool calls (information retrieval attempts)
    num_tool_calls = len(extract_information_blocks(solution_str))

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    # Calculate final reward (original logic)
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                final_reward = structure_format_score + retrieval_score # 0.3
            else:
                final_reward = structure_format_score # 0.2
        else:
            final_reward = 0
    else:
        if answer_correct:
            if is_valid_format:
                final_reward = score # 1
            else:
                final_reward = score - structure_format_score # 0.8
        elif is_valid_format:
            if retrieval_correct:
                final_reward = structure_format_score + retrieval_score # 0.3
            else:
                final_reward = structure_format_score # 0.2
        else:
            final_reward = final_format_score # 0.1

    # Return reward with detailed metrics for tracking
    # Note: The NaiveRewardManager wrapper expects a dict with "score" key
    # and automatically collects all keys as reward_extra_info
    return {
        "score": final_reward,  # Required: the actual reward value
        # Additional metrics (automatically collected by the wrapper)
        "format_valid": 1.0 if is_valid_format else 0.0,
        "has_answer": 1.0 if answer is not None else 0.0,
        "acc": 1.0 if answer_correct else 0.0,
        "num_tool_calls": float(num_tool_calls),
        "reward_final": final_reward,
    }
