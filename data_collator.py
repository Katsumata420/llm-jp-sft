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
        response_templates (Union(Dict[str, str], Dict[str, List[int]])): A dict of response templates.
            key: template name, value: token ids.
        instruction_templates (Optional[Union(Dict[str, str], Dict[str, List[int]]])): A dict of instruction templates.
            key: template name, value: token ids.
        mlm (bool): Whether to use masked language model.
            Note that this argument is not used in this class. It is just for compatibility.
        ignore_index (int): The token index to ignore when computing the loss.
    """

    def __init__(
        self,
        response_templates: Union[Dict[str, str], Dict[str, List[int]]],
        instruction_templates: Optional[Union[Dict[str, Optional[str]], Dict[str, Optional[List[int]]]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_templates = instruction_templates
        if isinstance(list(instruction_templates.values())[0], str):
            self.instructions_token_ids = {template_name: self.tokenizer.encode(template, add_special_tokens=False) for template_name, template in self.instruction_templates.items()}
            self.instructions_token_ids = {
                template_name: (
                    self.tokenizer.encode(template, add_special_tokens=False)
                    if template is not None
                    else None
                )
                for template_name, template in self.instruction_templates.items()
            }
        else:
            self.instructions_token_ids = instruction_templates

        self.response_templates = response_templates
        if isinstance(list(response_templates.values())[0], str):
            self.responses_token_ids = {template_name: self.tokenizer.encode(template, add_special_tokens=False) for template_name, template in self.response_templates.items()}
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
        assert isinstance(examples[0], dict), "DataCollatorForCompletionOnlyLMWithMultiTemplate requires a list of dictionaries as input"
        # hf model が使用する引数名は数年変化しないため、以下のように固定しても問題ない
        # https://github.com/huggingface/trl/blob/v0.8.1/trl/trainer/sft_trainer.py#L466
        model_used_columns = ["input_ids", "labels", "attention_mask"]
        examples_removed_unused_columns = [{k: v for k, v in example.items() if k in model_used_columns} for example in examples]
        batch = super().torch_call(examples_removed_unused_columns)

        if self.instruction_templates is None:
            for i in range(len(examples)):
                response_token_ids_start_idx: Optional[int] = None

                prompt_type = examples[i]["prompt_type"]
                if prompt_type == "none":
                    # if prompt_type is none, train the model as clm.
                    continue
                target_template = self.responses_token_ids.get(prompt_type)
                assert target_template is not None, f"Template '{prompt_type}' is not found"

                for idx in np.where(batch["labels"][i] == target_template[0])[0]:
                    if (
                        target_template
                        == batch["labels"][i][idx : idx + len(target_template)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_templates}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(target_template)

                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs: List[int] = []  # end idxs
                human_token_ids_idxs: List[int] = []  # start idxs

                prompt_type = examples[i]["prompt_type"]
                if prompt_type == "none":
                    # if prompt_type is none, train the model as clm.
                    continue
                target_template = self.responses_token_ids.get(prompt_type)
                assert target_template is not None, f"Template '{prompt_type}' is not found"

                # find response token ids
                for assistant_idx in np.where(batch["labels"][i] == target_template[0])[0]:
                    if (
                        target_template
                        == batch["labels"][i][assistant_idx : assistant_idx + len(target_template)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(target_template))


                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_templates}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                # instruction template に関しては、設定上 None が許容されるため、try-except で処理
                try:
                    target_template = self.instructions_token_ids[prompt_type]
                except KeyError:
                    raise KeyError(f"Template '{prompt_type}' is not found")

                # find instruction token ids
                if target_template is None:
                    warnings.warn(
                        f"Could not find prompt template '{prompt_type}'."
                        "This instance maybe single-turn conversation. "
                        "If you want to use multi-turn conversation, please set the prompt template."
                    )
                    human_token_ids_idxs.append(0)
                    assert len(human_token_ids_idxs) == len(
                        response_token_ids_idxs
                    ), "The number of instruction token ids and response token ids must be the same in the setting where the instruction template isn't found."
                else:
                    for human_idx in np.where(batch["labels"][i] == target_template[0])[0]:
                        if target_template == batch["labels"][i][human_idx : human_idx + len(target_template)].tolist():
                            human_token_ids_idxs.append(human_idx)

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
