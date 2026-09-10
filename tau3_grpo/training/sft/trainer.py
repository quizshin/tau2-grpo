"""SFT gradient accumulation for the pinned Transformers 5.5 / Accelerate stack."""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer


class Qwen35SFTTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        from peft import PeftModel
        from transformers import PreTrainedModel
        from transformers.trainer import TRAINING_ARGS_NAME

        if isinstance(self.model, PeftModel):
            return super()._save(output_dir, state_dict)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not isinstance(model, PreTrainedModel) or model.config.model_type != "qwen3_5":
            raise TypeError("Qwen35SFTTrainer requires a Qwen3.5 pretrained model")
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        # Transformers 5.5's reverse conversion rewrites Qwen3.5 checkpoint keys.
        # Trainer restores with load_state_dict, which does not undo that mapping.
        # Save native keys so best-checkpoint restore and vLLM see the same layout.
        model.save_pretrained(output, state_dict=state_dict, save_original_format=False)
        processing = self.processing_class or getattr(self.data_collator, "tokenizer", None)
        if processing is not None:
            processing.save_pretrained(output)
        torch.save(self.args, output / TRAINING_ARGS_NAME)

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        # Trainer 5.5.1 divides each micro-batch loss by its actual accumulation
        # window. Accelerate 1.12.0 backward otherwise divides it a second time.
        # Trainer controls micro-batch prefetching and sync boundaries itself;
        # keep that schedule in TrainingArguments and disable only the second divisor.
        self.accelerator.gradient_accumulation_steps = 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Keep the complete dialogue in the transformer. Project only positions
        # whose *next* token is supervised, avoiding enormous zero-gradient
        # vocabulary logits for tool observations and user/system messages.
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
        positions = torch.nonzero((shifted != -100).any(dim=0), as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("SFT batch contains no supervised next-token labels")
        model_inputs.update(logits_to_keep=positions, use_cache=False)
        outputs = model(**model_inputs)
        logits = outputs.logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               shifted[:, positions].reshape(-1), ignore_index=-100)
        return (loss, outputs) if return_outputs else loss
