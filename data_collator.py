import warnings
from typing import Union, List, Optional, Any, Dict

import numpy as np
from transformers import DataCollatorForLanguageModeling


class DataCollatorForCompletionOnlyLMWithMultiTemplate(DataCollatorForLanguageModeling):
    """Data collator used for completion tasks.

    This class is based on trl.DataCollatorForCompletionOnlyLM, but it supports multiple templates.
    The empty template is also supported.
    trl.DataCollatorForCompletionOnlyLM: https://github.com/huggingface/trl/blob/6c2f829bb7408660b0e581cde56fbff0980b9d7b/trl/trainer/utils.py#L69

    Args:
        response_templates (Union(List[str], List[List[int]])): A list of response templates.
        instruction_templates (Optional[Union(List[str], List[List[int]]])): A list of instruction templates.
        mlm (bool): Whether to use masked language model.
            Note that this argument is not used in this class. It is just for compatibility.
        ignore_index (int): The token index to ignore when computing the loss.
    """

    def __init__(
        self,
        response_templates: Union[List[str], List[List[int]]],
        instruction_templates: Optional[Union[List[str], List[List[int]]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_templates = instruction_templates
        if isinstance(instruction_templates[0], str):
            self.instructions_token_ids = [self.tokenizer.encode(template, add_special_tokens=False) for template in self.instruction_templates]
        else:
            self.instructions_token_ids = instruction_templates

        self.response_templates = response_templates
        if isinstance(response_templates[0], str):
            self.responses_token_ids = [self.tokenizer.encode(template, add_special_tokens=False) for template in self.response_templates]
        else:
            self.responses_token_ids = response_templates

        if not self.mlm and self.instruction_templates and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_templates is None:
            for i in range(len(examples)):
                response_token_ids_start_idx: Optional[int] = None

                for response_token_ids in self.responses_token_ids:
                    for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                        if (
                            response_token_ids
                            == batch["labels"][i][idx : idx + len(response_token_ids)].tolist()
                        ):
                            response_token_ids_start_idx = idx
                            break

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_templates}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs: List[int] = []  # end idxs
                human_token_ids_idxs: List[int] = []  # start idxs

                # find response token ids
                for response_token_ids in self.responses_token_ids:
                    for assistant_idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                        if (
                            response_token_ids
                            == batch["labels"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist()
                        ):
                            response_token_ids_idxs.append(assistant_idx + len(response_token_ids))

                    # break if we found the current response tokens
                    if len(response_token_ids_idxs) != 0:
                        break

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_templates}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                # find instruction token ids
                multi_human_token_ids = self.instructions_token_ids
                for human_token_ids in multi_human_token_ids:
                    for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                        if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                            human_token_ids_idxs.append(human_idx)

                    # break if we found the current instruction tokens
                    if len(human_token_ids_idxs) != 0:
                        break

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_templates}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch
