import json
import os
import sys
from loguru import logger
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count
from typing import Optional
from torch.utils.data import Dataset


def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count


def get_special_tokens():
    return ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


@dataclass
class ModelArgs:
    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    config: dict = field(default_factory=dict)
    cosine_schedule_num_cycles: float = 0.5
    custom_layer_parameters: list = field(default_factory=list)
    custom_parameter_groups: list = field(default_factory=list)
    dataloader_num_workers: int = 0
    do_lower_case: bool = False
    dynamic_quantize: bool = False
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_metric_minimize: bool = True
    early_stopping_patience: int = 3
    encoding: str = "utf-8"
    eval_batch_size: int = 8
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2000
    evaluate_during_training_verbose: bool = False
    evaluate_each_epoch: bool = True
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    local_rank: int = -1
    logging_steps: int = 50
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 128  # max length of input sequence
    model_name: str = None
    model_type: str = None
    multiprocessing_chunksize: int = -1
    n_gpu: int = 1
    no_cache: bool = False
    no_save: bool = False
    not_saved_args: list = field(default_factory=list)
    num_train_epochs: int = 1
    optimizer: str = "AdamW"
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = True
    polynomial_decay_schedule_lr_end: float = 1e-7
    polynomial_decay_schedule_power: float = 1.0
    process_count: int = field(default_factory=get_default_process_count)
    quantized_model: bool = False
    reprocess_input_data: bool = True
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = False
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 2000
    scheduler: str = "linear_schedule_with_warmup"
    silent: bool = False
    skip_special_tokens: bool = True
    tensorboard_dir: str = None
    thread_count: int = None
    tokenizer_name: str = None
    tokenizer_type: str = None
    train_batch_size: int = 8
    train_custom_parameters_only: bool = False
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_hf_datasets: bool = False
    use_multiprocessing: bool = False
    use_multiprocessing_for_evaluation: bool = False
    #     wandb_kwargs: dict = field(default_factory=dict)
    #     wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.0

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {key: value for key, value in asdict(self).items() if key not in self.not_saved_args}
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w", encoding='utf-8') as f:
            args_dict = self.get_args_for_saving()
            if args_dict['dataset_class'] is not None and not isinstance(args_dict["dataset_class"], str):
                args_dict['dataset_class'] = type(args_dict['dataset_class']).__name__
            if args_dict["tokenizer_type"] is not None and not isinstance(args_dict["tokenizer_type"], str):
                args_dict["tokenizer_type"] = type(args_dict["tokenizer_type"]).__name__
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r", encoding='utf-8') as f:
                    model_args = json.load(f)
                if model_args["dataset_class"]:
                    logger.warning(
                        "This model was trained using a custom dataset_class."
                        "This cannot be loaded automatically and must be specified in the model args"
                        "when loading the model."
                    )
                self.update_from_dict(model_args)


@dataclass
class ChatGlmArgs(ModelArgs):
    """
    Model args for a ChatGLMModel
    """
    model_class: str = "ChatGlmArgs"
    dataset_class: Dataset = None
    learning_rate: float = 2e-4
    fp16: bool = True
    int8: bool = False
    quantization_bit: int = None  # if use quantization bit, set 4, else None
    debug: bool = False
    max_seq_length: int = 256  # max length of input sequence
    max_length = 384  # max length of the sequence to be generated
    max_new_tokens = 384
    do_sample: bool = True
    early_stopping: bool = True
    evaluate_generated_text: bool = True
    optimizer: str = "adamw_torch"
    save_strategy: str = "steps"
    evaluation_strategy: str = "no"
    eval_steps: int = None
    save_steps: int = 400
    max_eval_samples: int = 20
    length_penalty: float = 2.0
    num_beams: int = 1
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 0.95
    special_tokens_list: list = field(default_factory=list)
    top_k: float = None
    top_p: float = 0.7
    model_name_or_path: Optional[str] = field(default="THUDM/chatglm-6b")
    dataset_name_or_path: Optional[str] = field(default="shibing624/alpaca-zh")
    use_lora: bool = True
    lora_name: str = field(default="adapter_model.bin")
    lora_r: int = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["query_key_value"]
    lora_bias = "none"
    only_lora_state_dict: bool = False
    num_train_epochs = 1
    max_steps = -1
    per_device_train_batch_size = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps = 4
    save_total_limit = 3
    remove_unused_columns = False
    logging_steps = 50
    resume_from_checkpoint: str = None
    no_repeat_ngram_size = 0

    encoder_no_repeat_ngram_size = 0
    bad_words_ids = None
    eos_token_id = 130005
    min_length = 10
    min_new_tokens = None
    forced_bos_token_id = None
    forced_eos_token_id = 130005
    encoder_repetition_penalty: float = 1.0
    diversity_penalty = 0.0
    remove_invalid_values = False
    exponential_decay_length_penalty = None
    suppress_tokens = None
    begin_suppress_tokens = None
    forced_decoder_ids = None
    renormalize_logits = None
    max_time = None
    typical_p = 0.2
    epsilon_cutoff = None
    eta_cutoff = None