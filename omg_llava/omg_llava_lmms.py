from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Tuple
from tqdm import tqdm
import logging
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
import torch
import warnings
from xtuner.dataset.utils import expand2square
from xtuner.registry import BUILDER
from xtuner.tools.utils import get_stop_criteria
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.model.utils import prepare_inputs_labels_for_multimodal

from mmengine.config import Config


warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")


class OMGLLaVA(lmms):
    def __init__(
        self,
        config,
        pretrained,
        prompt_template='internlm2_chat',
        model_id='maverickrzw/PixelLM-7B',
        device: str = "cuda",
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = torch.device(device)

        cfg = Config.fromfile(config)
        model_name = cfg.model.type if isinstance(cfg.model.type,
                                                  str) else cfg.model.type.__name__
        if 'LLaVAModel' or 'OMG' in model_name:
            cfg.model.pretrained_pth = None

        self._model = BUILDER.build(cfg.model)
        self._model.load_state_dict(guess_load_checkpoint(pretrained), strict=False)
        print(f'Load PTH model from {pretrained}')

        image_processor = cfg.image_processor
        image_processor_type = image_processor.pop('type')
        self._image_processor = image_processor_type(**image_processor)

        self._tokenizer = self._model.tokenizer
        self._model.eval()
        self._model.llm.eval()
        self._model.visual_encoder.eval()
        self._model.projector.eval()

        self.prompt_template = PROMPT_TEMPLATE[prompt_template]

        self.stop_criteria = get_stop_criteria(
            tokenizer=self._tokenizer, stop_words=self.prompt_template.get('STOP_WORDS', [])
        )

        self._config = self._model.llm.config
        self.batch_size_per_gpu = int(batch_size)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    @property
    def projector(self):
        return self.model.projector

    @property
    def visual_encoder(self):
        return self.model.visual_encoder

    @property
    def llm(self):
        return self.model.llm

    @torch.no_grad()
    def predict(self, prompt, image, gen_kwargs):
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt, round=1)
        image = image.convert('RGB')

        image = expand2square(
            image, tuple(int(x * 255) for x in self._image_processor.image_mean))
        image = self._image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]

        image = image[None].to(dtype=self.visual_encoder.dtype, device=self.device)

        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        if isinstance(visual_outputs, list) or isinstance(visual_outputs, tuple)\
                or isinstance(visual_outputs, torch.Tensor):
            pixel_values = self.projector(visual_outputs)
        else:
            pixel_values = self.projector(
                visual_outputs.hidden_states[-2][:, 1:])

        chunk_encode = []
        for idx, chunk in enumerate(prompt.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).to(self._device).unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

        generate_output = self.llm.generate(
            **mm_inputs,
            max_new_tokens=gen_kwargs.get('max_new_tokens', 100),
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)

        text_output = self.tokenizer.decode(
            generate_output[0], skip_special_tokens=True).strip()

        import pdb; pdb.set_trace()

        return text_output

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # import pdb; pdb.set_trace()
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            assert len(visuals) == 1, f"Currently only support 1 image: {contexts}, {visuals}"
            text_output = self.predict(contexts, visuals[0], gen_kwargs)
            # import pdb; pdb.set_trace()
            res.append(text_output)
            pbar.update(1)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "lisa not support"
