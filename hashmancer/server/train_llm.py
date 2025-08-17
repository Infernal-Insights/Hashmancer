from __future__ import annotations

"""Lightweight helpers for fine-tuning a local language model."""

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
)


def train_model(
    dataset: Path,
    model_path: str,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
) -> None:
    """Fine-tune *model_path* on *dataset* and save to *output_dir*."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    train_data = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(dataset),
        block_size=128,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        data_collator=collator,
    )

    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
