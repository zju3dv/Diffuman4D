import torch
from tqdm import tqdm
from threading import Lock
from functools import partial
from collections import defaultdict

from src.data.spatem_dataset import SpaTemDataset
from src.diffusers.pipelines.diffuman4d.pipeline_diffuman4d import Diffuman4DPipeline
from src.samplers.utils.sampling_utils import save_sampling_results, check_sampling_results
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SlidingIterativeSampler:
    def __init__(
        self,
        dataset: SpaTemDataset,
        pipelines: list[Diffuman4DPipeline],
        output_dir: str = "./results/debug",
        # denoising args
        window_size: int = 12,  # number of target samples in the window
        sliding_stride: int = 1,  # sliding stride of the window
        sliding_shift: int = 0,  # shift of starting position of the window
        bidirectional: bool = True,  # whether to slide in both directions
        num_denoising_steps: int = 1,  # number of denoising steps for each window
        alternation_rounds: int = 3,  # number of alternation rounds (spatial -> temporal -> spatial -> ...)
        guidance_scale: float = 2.0,
        # sampling range args
        spa_label_range: list[int, int, int] = [0, 48, 1],
        tem_label_range: list[int, int, int] = [0, 150, 1],
        spa_labels: list[int, ...] = None,
        tem_labels: list[int, ...] = None,
        input_spa_labels: list[int, ...] = [1, 13, 25, 37],
    ):
        self.dataset = dataset
        self.pipelines = pipelines
        self.output_dir = output_dir
        # denoising args
        self.window_size = window_size
        self.sliding_stride = sliding_stride
        self.sliding_shift = sliding_shift
        self.bidirectional = bidirectional
        self.num_denoising_steps = num_denoising_steps
        self.alternation_rounds = alternation_rounds
        self.guidance_scale = guidance_scale

        # sampling range args
        if spa_labels is not None:
            self.spa_labels = [f"{int(i):02d}" for i in spa_labels]
        elif spa_label_range is not None:
            b, e, s = spa_label_range
            self.spa_labels = [f"{int(i):02d}" for i in range(b, e, s)]
        else:
            raise ValueError("spa_labels or spa_label_range must be provided")

        if tem_labels is not None:
            self.tem_labels = [f"{int(i):06d}" for i in tem_labels]
        elif tem_label_range is not None:
            b, e, s = tem_label_range
            self.tem_labels = [f"{int(i):06d}" for i in range(b, e, s)]
        else:
            raise ValueError("tem_labels or tem_label_range must be provided")

        self.input_spa_labels = [f"{int(i):02d}" for i in input_spa_labels]
        self.target_spa_labels = [label for label in self.spa_labels if label not in self.input_spa_labels]
        log.info(
            f"Found {len(self.spa_labels)} spatial labels, {len(self.input_spa_labels)} input spatial labels, {len(self.tem_labels)} temporal labels."
        )

        if self.window_size > len(self.target_spa_labels):
            raise ValueError(
                f"window_size(={self.window_size}) must be <= len(target_spa_labels)(={len(self.target_spa_labels)})"
            )

        if len(self.target_spa_labels) % self.sliding_stride != 0:
            raise ValueError(
                f"len(target_spa_labels)(={len(self.target_spa_labels)}) % sliding_stride(={self.sliding_stride}) must be 0"
            )
        if len(self.tem_labels) % self.sliding_stride != 0:
            raise ValueError(
                f"len(tem_labels)(={len(self.tem_labels)}) % sliding_stride(={self.sliding_stride}) must be 0"
            )

        if self.alternation_rounds > 1 and self.window_size > len(self.tem_labels):
            raise ValueError(
                f"window_size(={self.window_size}) must be <= the number of tem_labels(={len(self.tem_labels)}) when alternation_rounds > 1"
            )

        # spatio-temporal latents grid
        self.latents = defaultdict(dict)
        self.timestep_indices = defaultdict(dict)
        for spa_label in self.spa_labels:
            for tem_label in self.tem_labels:
                self.latents[spa_label][tem_label] = None
                self.timestep_indices[spa_label][tem_label] = 0
        self.lock = Lock()  # multi-threading lock

        # prepare task lists
        self.prepare_tasks()

    def load_sample(self, alt: int, domain: str, domain_label: str):
        def ref_indices(all_labels, ref_labels):
            return [all_labels.index(label) for label in ref_labels]

        # prepare input and target labels and indices
        if domain == "spatial":
            spa_labels = self.spa_labels
            tem_labels = [domain_label]
            input_indices = torch.tensor(ref_indices(self.spa_labels, self.input_spa_labels))
            target_indices = torch.tensor(ref_indices(self.spa_labels, self.target_spa_labels))
        elif domain == "temporal":
            spa_labels = [domain_label]
            tem_labels = self.tem_labels
            num_frames_half = len(self.tem_labels)
            # first half is input, second half is target
            input_indices = torch.tensor(list(range(num_frames_half)))
            target_indices = torch.tensor(list(range(num_frames_half, 2 * num_frames_half)))

        # load sample data
        sample = self.dataset.get_item(
            scene_label=self.dataset.scene_label,
            spa_labels=spa_labels,
            tem_labels=tem_labels,
            input_spa_labels=self.input_spa_labels,
        )
        sample["alt"] = alt
        sample["domain"] = domain
        sample["domain_label"] = domain_label
        sample["input_indices"] = input_indices
        sample["target_indices"] = target_indices

        # set conditional masks
        def set_cond_masks(cond_masks, input_indices):
            cond_masks[...] = 1.0
            cond_masks[input_indices, ...] = 0.0
            return cond_masks

        sample["cond_masks"] = set_cond_masks(sample["cond_masks"], input_indices)

        # prepare latents and timestep indices
        with self.lock:
            latents = []
            timestep_indices = []
            for _, spa_label, tem_label in sample["labels"]:
                latents.append(self.latents[spa_label][tem_label])
                timestep_indices.append(self.timestep_indices[spa_label][tem_label])

        timestep_indices = torch.tensor(timestep_indices)
        sample["latents"] = None if timestep_indices[target_indices[0]] == 0 else torch.stack(latents, dim=0)
        sample["timestep_indices"] = timestep_indices

        return sample

    @torch.no_grad()
    def denoise(self, sample: dict, pipe_idx: int = 0):
        pipeline = self.pipelines[pipe_idx]
        task_label = f"alt{sample['alt']}_{'spa' if sample['domain'] == 'temporal' else 'tem'}{sample['domain_label']}"

        # denoise a spatial or temporal sample sequence
        result = pipeline.sliding_iterative_denoise(
            pixel_values=sample["pixel_values"],
            plucker_embeds=sample["plucker_embeds"],
            skeletons=sample["skeletons"],
            cond_masks=sample["cond_masks"],
            latents=sample["latents"],
            domain=sample["domain"],
            timestep_indices=sample["timestep_indices"],
            # denoising args
            window_size=self.window_size,
            sliding_stride=self.sliding_stride,
            sliding_shift=self.sliding_shift,
            bidirectional=self.bidirectional,
            num_denoising_steps=self.num_denoising_steps,
            alternation_rounds=self.alternation_rounds,
            guidance_scale=self.guidance_scale,
            tqdm=partial(tqdm, desc=f"Denoising {task_label} on {pipeline.device}"),
        )

        # update latents and timestep indices
        with self.lock:
            for label, latent, timestep_index in zip(sample["labels"], result["latents"], result["timestep_indices"]):
                _, spa_label, tem_label = label
                self.latents[spa_label][tem_label] = latent.cpu()
                self.timestep_indices[spa_label][tem_label] = timestep_index.item()

        sample["images"] = result["images"].float().cpu()
        sample["timestep_indices"] = result["timestep_indices"].cpu()
        sample["fully_denoised"] = result["fully_denoised"].cpu()
        return sample

    def prepare_tasks(self):
        domains = (["spatial", "temporal"] * self.alternation_rounds)[: self.alternation_rounds]
        self.all_tasks = []

        for i, domain in enumerate(domains):
            domain_labels = self.tem_labels if domain == "spatial" else self.target_spa_labels
            tasks = [{"alt": i + 1, "domain": domain, "domain_label": domain_label} for domain_label in domain_labels]
            self.all_tasks.append(tasks)

    def execute_one_task(self, task: dict, pipe_idx: int = 0):
        sample = self.load_sample(**task)
        sample = self.denoise(sample, pipe_idx=pipe_idx)
        save_sampling_results(sample, output_dir=self.output_dir)

    def execute_tasks(self):
        for tasks in self.all_tasks:
            for task in tasks:
                self.execute_one_task(task)

        if not check_sampling_results(self.spa_labels, self.tem_labels, self.output_dir):
            raise ValueError("Sampling failed.")
