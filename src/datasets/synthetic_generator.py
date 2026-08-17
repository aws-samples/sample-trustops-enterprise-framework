"""
Synthetic data generator for dataset augmentation.

This module provides functionality to generate synthetic examples by
paraphrasing existing examples using LLMs. This is useful for increasing
dataset size and diversity.

Requirement 2.16: Implement synthetic data generator
"""

import asyncio
import json
from typing import Any, Optional

from src.adapters.base_adapter import InferenceRequest
from src.clients.inference_client import InferenceClient
from src.data_models.dataset import DatasetTaskType


class SyntheticDataGenerator:
    """Generates synthetic data for dataset augmentation using LLMs."""

    def __init__(
        self,
        inference_client: InferenceClient,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    ):
        """
        Initialize the synthetic data generator.

        Args:
            inference_client: InferenceClient for invoking LLMs
            model_id: Model to use for paraphrasing (default: Claude 3 Haiku)
        """
        self.inference_client = inference_client
        self.model_id = model_id

    async def generate_synthetic(
        self,
        examples: list[dict[str, Any]],
        augmentation_factor: float = 1.5,
        strategy: str = "paraphrase",
        task_type: Optional[DatasetTaskType] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate synthetic examples for augmentation.

        Args:
            examples: Original dataset examples
            augmentation_factor: Multiplier for dataset size (e.g., 1.5 = 50% more examples)
            strategy: Augmentation strategy ("paraphrase", "diverse", "creative")
            task_type: Optional task type for context-aware generation

        Returns:
            List of synthetic examples

        Raises:
            ValueError: If augmentation_factor < 1.0 or examples is empty
        """
        if not examples:
            raise ValueError("Cannot generate synthetic data from empty dataset")

        if augmentation_factor < 1.0:
            raise ValueError(
                f"Augmentation factor must be >= 1.0, got {augmentation_factor}"
            )

        # Calculate number of synthetic examples to generate
        num_synthetic = int(len(examples) * (augmentation_factor - 1.0))

        if num_synthetic == 0:
            return []

        # Select examples to augment (cycle through if needed)
        examples_to_augment = []
        for i in range(num_synthetic):
            example_idx = i % len(examples)
            examples_to_augment.append(examples[example_idx])

        # Generate synthetic examples based on strategy
        if strategy == "paraphrase":
            synthetic = await self._generate_paraphrased(
                examples_to_augment, task_type
            )
        elif strategy == "diverse":
            synthetic = await self._generate_diverse(
                examples_to_augment, task_type
            )
        elif strategy == "creative":
            synthetic = await self._generate_creative(
                examples_to_augment, task_type
            )
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Supported: paraphrase, diverse, creative"
            )

        return synthetic

    async def _generate_paraphrased(
        self,
        examples: list[dict[str, Any]],
        task_type: Optional[DatasetTaskType] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate paraphrased versions of examples.

        Uses simple paraphrasing to maintain semantic meaning while
        varying the expression.

        Args:
            examples: Examples to paraphrase
            task_type: Optional task type for context

        Returns:
            List of paraphrased examples
        """
        synthetic = []

        for example in examples:
            try:
                paraphrased = await self._paraphrase_example(example, task_type)
                if paraphrased:
                    synthetic.append(paraphrased)
            except Exception as e:
                # Log error but continue with other examples
                print(f"Warning: Failed to paraphrase example: {str(e)}")
                continue

        return synthetic

    async def _generate_diverse(
        self,
        examples: list[dict[str, Any]],
        task_type: Optional[DatasetTaskType] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate diverse variations of examples.

        Creates variations with different phrasings, perspectives,
        and formality levels.

        Args:
            examples: Examples to vary
            task_type: Optional task type for context

        Returns:
            List of diverse examples
        """
        synthetic = []

        for example in examples:
            try:
                varied = await self._create_diverse_variation(example, task_type)
                if varied:
                    synthetic.append(varied)
            except Exception as e:
                print(f"Warning: Failed to create diverse variation: {str(e)}")
                continue

        return synthetic

    async def _generate_creative(
        self,
        examples: list[dict[str, Any]],
        task_type: Optional[DatasetTaskType] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate creative variations of examples.

        Creates more substantial variations while maintaining
        the core task and expected output.

        Args:
            examples: Examples to vary
            task_type: Optional task type for context

        Returns:
            List of creative examples
        """
        synthetic = []

        for example in examples:
            try:
                creative = await self._create_creative_variation(example, task_type)
                if creative:
                    synthetic.append(creative)
            except Exception as e:
                print(f"Warning: Failed to create creative variation: {str(e)}")
                continue

        return synthetic

    async def _paraphrase_example(
        self,
        example: dict[str, Any],
        task_type: Optional[DatasetTaskType] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Paraphrase a single example.

        Args:
            example: Example to paraphrase
            task_type: Optional task type for context

        Returns:
            Paraphrased example or None if failed
        """
        # Extract prompt and completion
        prompt_text = self._extract_field(
            example, ["prompt", "question", "text", "input"]
        )
        completion_text = self._extract_field(
            example, ["completion", "answer", "response", "output"]
        )

        if not prompt_text:
            return None

        # Build paraphrasing prompt
        system_prompt = self._build_paraphrase_prompt(task_type)

        user_prompt = f"""Paraphrase the following text while maintaining its exact meaning and intent:

Original: {prompt_text}

Paraphrased:"""

        # Invoke LLM
        request = InferenceRequest(
            prompt=f"{system_prompt}\n\n{user_prompt}",
            max_tokens=len(prompt_text.split()) * 2 + 50,
            temperature=0.7,
            top_p=0.9,
        )

        try:
            response = await self.inference_client.invoke(
                self.model_id, request, timeout_seconds=30.0
            )

            paraphrased_prompt = response.text.strip()

            # Create new example with paraphrased prompt
            new_example = example.copy()

            # Update the prompt field
            if "prompt" in example:
                new_example["prompt"] = paraphrased_prompt
            elif "question" in example:
                new_example["question"] = paraphrased_prompt
            elif "text" in example:
                new_example["text"] = paraphrased_prompt
            elif "input" in example:
                new_example["input"] = paraphrased_prompt

            # Optionally paraphrase completion if present
            if completion_text:
                completion_paraphrased = await self._paraphrase_text(
                    completion_text, system_prompt
                )
                if completion_paraphrased:
                    if "completion" in example:
                        new_example["completion"] = completion_paraphrased
                    elif "answer" in example:
                        new_example["answer"] = completion_paraphrased
                    elif "response" in example:
                        new_example["response"] = completion_paraphrased
                    elif "output" in example:
                        new_example["output"] = completion_paraphrased

            # Mark as synthetic
            new_example["_synthetic"] = True
            new_example["_augmentation_strategy"] = "paraphrase"

            return new_example

        except Exception as e:
            print(f"Error paraphrasing: {str(e)}")
            return None

    async def _paraphrase_text(
        self, text: str, system_prompt: str
    ) -> Optional[str]:
        """
        Paraphrase a text string.

        Args:
            text: Text to paraphrase
            system_prompt: System prompt for context

        Returns:
            Paraphrased text or None if failed
        """
        user_prompt = f"""Paraphrase the following text while maintaining its exact meaning:

Original: {text}

Paraphrased:"""

        request = InferenceRequest(
            prompt=f"{system_prompt}\n\n{user_prompt}",
            max_tokens=len(text.split()) * 2 + 50,
            temperature=0.7,
            top_p=0.9,
        )

        try:
            response = await self.inference_client.invoke(
                self.model_id, request, timeout_seconds=30.0
            )
            return response.text.strip()
        except Exception:
            return None

    async def _create_diverse_variation(
        self,
        example: dict[str, Any],
        task_type: Optional[DatasetTaskType] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Create a diverse variation of an example.

        Args:
            example: Example to vary
            task_type: Optional task type for context

        Returns:
            Diverse variation or None if failed
        """
        prompt_text = self._extract_field(
            example, ["prompt", "question", "text", "input"]
        )

        if not prompt_text:
            return None

        system_prompt = self._build_diverse_prompt(task_type)

        user_prompt = f"""Create a diverse variation of the following text by changing the phrasing, perspective, or formality level while maintaining the core meaning:

Original: {prompt_text}

Variation:"""

        request = InferenceRequest(
            prompt=f"{system_prompt}\n\n{user_prompt}",
            max_tokens=len(prompt_text.split()) * 2 + 50,
            temperature=0.8,
            top_p=0.9,
        )

        try:
            response = await self.inference_client.invoke(
                self.model_id, request, timeout_seconds=30.0
            )

            varied_prompt = response.text.strip()

            new_example = example.copy()

            # Update the prompt field
            if "prompt" in example:
                new_example["prompt"] = varied_prompt
            elif "question" in example:
                new_example["question"] = varied_prompt
            elif "text" in example:
                new_example["text"] = varied_prompt
            elif "input" in example:
                new_example["input"] = varied_prompt

            new_example["_synthetic"] = True
            new_example["_augmentation_strategy"] = "diverse"

            return new_example

        except Exception as e:
            print(f"Error creating diverse variation: {str(e)}")
            return None

    async def _create_creative_variation(
        self,
        example: dict[str, Any],
        task_type: Optional[DatasetTaskType] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Create a creative variation of an example.

        Args:
            example: Example to vary
            task_type: Optional task type for context

        Returns:
            Creative variation or None if failed
        """
        prompt_text = self._extract_field(
            example, ["prompt", "question", "text", "input"]
        )
        completion_text = self._extract_field(
            example, ["completion", "answer", "response", "output"]
        )

        if not prompt_text:
            return None

        system_prompt = self._build_creative_prompt(task_type)

        # Include completion as context if available
        context = f"\nExpected type of response: {completion_text}" if completion_text else ""

        user_prompt = f"""Create a creative variation of the following text that maintains the same task and intent but uses different scenarios, examples, or contexts:{context}

Original: {prompt_text}

Creative variation:"""

        request = InferenceRequest(
            prompt=f"{system_prompt}\n\n{user_prompt}",
            max_tokens=len(prompt_text.split()) * 2 + 100,
            temperature=0.9,
            top_p=0.95,
        )

        try:
            response = await self.inference_client.invoke(
                self.model_id, request, timeout_seconds=30.0
            )

            creative_prompt = response.text.strip()

            new_example = example.copy()

            # Update the prompt field
            if "prompt" in example:
                new_example["prompt"] = creative_prompt
            elif "question" in example:
                new_example["question"] = creative_prompt
            elif "text" in example:
                new_example["text"] = creative_prompt
            elif "input" in example:
                new_example["input"] = creative_prompt

            new_example["_synthetic"] = True
            new_example["_augmentation_strategy"] = "creative"

            return new_example

        except Exception as e:
            print(f"Error creating creative variation: {str(e)}")
            return None

    def _extract_field(
        self, example: dict[str, Any], field_names: list[str]
    ) -> Optional[str]:
        """
        Extract a field value from an example.

        Args:
            example: Example dictionary
            field_names: List of possible field names to try

        Returns:
            Field value as string or None if not found
        """
        for field in field_names:
            if field in example and example[field]:
                return str(example[field])
        return None

    def _build_paraphrase_prompt(
        self, task_type: Optional[DatasetTaskType] = None
    ) -> str:
        """
        Build system prompt for paraphrasing.

        Args:
            task_type: Optional task type for context

        Returns:
            System prompt string
        """
        base_prompt = "You are a helpful assistant that paraphrases text while maintaining exact meaning and intent."

        if task_type == DatasetTaskType.QA:
            return f"{base_prompt} Focus on rephrasing questions clearly."
        elif task_type == DatasetTaskType.SUMMARIZATION:
            return f"{base_prompt} Focus on rephrasing text to summarize."
        elif task_type == DatasetTaskType.CLASSIFICATION:
            return f"{base_prompt} Focus on rephrasing text for classification tasks."
        elif task_type == DatasetTaskType.TEXT_GENERATION:
            return f"{base_prompt} Focus on rephrasing prompts for text generation."
        elif task_type == DatasetTaskType.CHAT:
            return f"{base_prompt} Focus on rephrasing conversational messages."
        else:
            return base_prompt

    def _build_diverse_prompt(
        self, task_type: Optional[DatasetTaskType] = None
    ) -> str:
        """
        Build system prompt for diverse variations.

        Args:
            task_type: Optional task type for context

        Returns:
            System prompt string
        """
        return (
            "You are a helpful assistant that creates diverse variations of text "
            "by changing phrasing, perspective, or formality while maintaining "
            "the core meaning and intent."
        )

    def _build_creative_prompt(
        self, task_type: Optional[DatasetTaskType] = None
    ) -> str:
        """
        Build system prompt for creative variations.

        Args:
            task_type: Optional task type for context

        Returns:
            System prompt string
        """
        return (
            "You are a helpful assistant that creates creative variations of text "
            "using different scenarios, examples, or contexts while maintaining "
            "the same task type and expected output format."
        )


async def generate_synthetic_data(
    examples: list[dict[str, Any]],
    inference_client: InferenceClient,
    augmentation_factor: float = 1.5,
    strategy: str = "paraphrase",
    task_type: Optional[DatasetTaskType] = None,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
) -> list[dict[str, Any]]:
    """
    Convenience function to generate synthetic data.

    Args:
        examples: Original dataset examples
        inference_client: InferenceClient for invoking LLMs
        augmentation_factor: Multiplier for dataset size (default: 1.5)
        strategy: Augmentation strategy (default: "paraphrase")
        task_type: Optional task type for context
        model_id: Model to use for generation (default: Claude 3 Haiku)

    Returns:
        List of synthetic examples

    Raises:
        ValueError: If augmentation_factor < 1.0 or examples is empty
    """
    generator = SyntheticDataGenerator(inference_client, model_id)
    return await generator.generate_synthetic(
        examples, augmentation_factor, strategy, task_type
    )
