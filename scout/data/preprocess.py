# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os
import sys

import datasets
import pandas as pd

from scout.data.templates import get_template, list_templates
from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_single_row(row, current_split_name, row_index, system_content, user_content_prefix):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame
        system_content: System message content
        user_content_prefix: User message prefix before the question

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question", "").strip()

    # Build prompt structure
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    # Extract ground truth from golden_answers (FlashRAG format)
    golden_answers = row.get("golden_answers", [])
    ground_truth = {"target": golden_answers}

    # Create reward_model structure
    reward_model_data = {
        "style": "rule",
        "ground_truth": ground_truth
    }

    # Process data source
    data_source_tagged = "searchR1_" + str(row.get("data_source", ""))

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    # Extract question type for ParallelSearch reward function
    # Check metadata first, then row-level type field, default to "na"
    metadata = row.get("metadata")
    if metadata and isinstance(metadata, dict) and "type" in metadata:
        query_type = metadata.get("type")
    elif "type" in row:
        query_type = row.get("type")
    else:
        # Default to "na" if type is not found
        query_type = "na"

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
        "type": query_type,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": "fact-reasoning",
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Load template or use direct overrides
    if args.system_content or args.user_content_prefix:
        # Use direct content overrides if provided
        if not args.system_content or not args.user_content_prefix:
            logger.error("Both --system_content and --user_content_prefix must be provided when using direct overrides")
            sys.exit(1)
        system_content = args.system_content
        user_content_prefix = args.user_content_prefix
        logger.info("Using directly provided system and user content")
    else:
        # Load template
        try:
            template = get_template(args.template)
            system_content = template.system_content
            user_content_prefix = template.user_content_prefix
            logger.info(f"Using template: {args.template} - {template.description}")
        except KeyError as e:
            logger.error(str(e))
            sys.exit(1)

    # Process train and test datasets
    data_sources_train = args.train_data_sources.split(',') if args.train_data_sources else []
    data_sources_test = args.test_data_sources.split(',') if args.test_data_sources else []

    for split, data_sources in [("train", data_sources_train), ("test", data_sources_test)]:
        if not data_sources:
            logger.warning(f"No data sources specified for {split} split, skipping...")
            continue

        logger.info(f"Processing {split} split with data sources: {data_sources}")
        all_datasets = []

        for data_source in data_sources:
            try:
                logger.info(f"Loading {data_source} from RUC-NLPIR/FlashRAG_datasets...")
                dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

                # Select appropriate split
                if split == "train":
                    if 'train' in dataset:
                        source_dataset = dataset['train']
                    else:
                        logger.warning(f"No train split for {data_source}, skipping...")
                        continue
                else:  # test
                    if 'test' in dataset:
                        source_dataset = dataset['test']
                    elif 'dev' in dataset:
                        logger.info(f"Using dev split for {data_source}")
                        source_dataset = dataset['dev']
                    else:
                        logger.warning(f"No test/dev split for {data_source}, skipping...")
                        continue

                # Convert to pandas DataFrame for processing
                df_raw = source_dataset.to_pandas()
                df_raw['data_source'] = data_source  # Add data_source column

                logger.info(f"Loaded {len(df_raw)} rows from {data_source}")

                # Process each row
                def apply_process_row(row):
                    return process_single_row(
                        row,
                        current_split_name=split,
                        row_index=row.name,
                        system_content=system_content,
                        user_content_prefix=user_content_prefix,
                    )

                df_processed = df_raw.apply(apply_process_row, axis=1)
                all_datasets.append(df_processed)

            except Exception as e:
                logger.error(f"Error loading {data_source}: {e}")
                continue

        if all_datasets:
            # Concatenate all datasets for this split
            combined_df = pd.concat(all_datasets, ignore_index=True)
            output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
            combined_df.to_parquet(output_file_path, index=False)
            logger.info(f"Saved {len(combined_df)} processed rows to {output_file_path}")
        else:
            logger.warning(f"No datasets processed for {split} split")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process QA datasets from RUC-NLPIR/FlashRAG_datasets to Search-R1 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train_data_sources",
        default="nq,hotpotqa",
        help="Comma-separated list of data sources for training (e.g., 'nq,hotpotqa')",
    )
    parser.add_argument(
        "--test_data_sources",
        default="nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle",
        help="Comma-separated list of data sources for testing",
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/searchR1_processed",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    # Template configuration
    parser.add_argument(
        "--template",
        default="search_r1",
        help="Prompt template to use (default: search_r1). Use --list_templates to see available templates.",
    )
    parser.add_argument(
        "--list_templates", action="store_true", help="List all available prompt templates and exit."
    )

    # Direct content overrides (optional, overrides template selection)
    parser.add_argument("--system_content", default=None, help="Override system message content directly.")
    parser.add_argument("--user_content_prefix", default=None, help="Override user message prefix directly.")

    args = parser.parse_args()

    # Handle --list_templates
    if args.list_templates:
        templates = list_templates()
        print("Available prompt templates:")
        print("-" * 80)
        for name, description in templates.items():
            print(f"  {name:20s} - {description}")
        print("-" * 80)
        sys.exit(0)

    main()
