#!/usr/bin/env python3
"""
Script to load and save Qwen 2.5 VL 3B tokenizer/preprocessing components
"""

import argparse
import os
import json
from transformers import AutoTokenizer, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description='Convert Jinja chat template to JSON')
    parser.add_argument('--template-file',
                        default='qwen_chat_template_tool.jinja',
                        help='Path to the Jinja template file')
    parser.add_argument('--output-dir',
                        default='qwen_2p5_VL_tool_tokenizer',
                        help='Directory to save the chat_template.json file')

    args = parser.parse_args()

    try:
        # Read the Jinja template file
        print(f"Reading template file: {args.template_file}")
        with open(args.template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # Create the JSON structure
        chat_template_data = {
            "chat_template": template_content
        }

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Write to JSON file
        output_path = os.path.join(args.output_dir, 'chat_template.json')
        print(f"Writing JSON to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chat_template_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully saved chat template to '{output_path}'")

    except FileNotFoundError:
        print(f"❌ Error: Template file '{args.template_file}' not found")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


def copy_tokenizer():
    parser = argparse.ArgumentParser(description='Load and save Qwen 2.5 VL 3B preprocessing components')
    parser.add_argument('--model-name',
                        default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Model name from Hugging Face Hub')
    parser.add_argument('--save-path',
                        default='qwen_2p5_VL_tool_tokenizer',
                        help='Directory to save the preprocessing components')
    parser.add_argument('--cache-dir',
                        help='Cache directory for downloading models (optional)')

    args = parser.parse_args()

    print(f"Loading preprocessing components for {args.model_name}...")

    try:
        # Load the tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )

        # Load the processor (for vision-language models)
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )

        # Create save directory if it doesn't exist
        os.makedirs(args.save_path, exist_ok=True)

        # Save tokenizer
        print(f"Saving tokenizer to {args.save_path}...")
        tokenizer.save_pretrained(args.save_path)

        # Save processor
        print(f"Saving processor to {args.save_path}...")
        processor.save_pretrained(args.save_path)

        print(f"✅ Successfully saved preprocessing components to '{args.save_path}'")
        print(f"📁 Contents saved:")
        print(f"   - Tokenizer configuration and vocabulary")
        print(f"   - Image processor configuration")
        print(f"   - All necessary preprocessing files")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())