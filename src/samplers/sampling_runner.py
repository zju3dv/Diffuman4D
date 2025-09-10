from queue import Queue, Empty
from threading import Thread

from src.samplers.sliding_iterative_sampler import SlidingIterativeSampler
from src.samplers.utils.sampling_utils import check_sampling_results
from src.data.utils.metric_utils import evaluate_results
from src.utils import RankedLogger

from scripts.nerfstudio.diffuman4d_to_nerfstudio import diffuman4d_to_nerfstudio

log = RankedLogger(__name__, rank_zero_only=True)


class SamplingRunner:
    def __init__(self, sampler: SlidingIterativeSampler):
        self.sampler = sampler

    def prepare_task_queues(self):
        self.task_queues = []
        for tasks in self.sampler.all_tasks:
            task_queue = Queue()
            for task in tasks:
                task_queue.put(task)
            self.task_queues.append(task_queue)

    def parallel_execute_tasks(self, task_queue: Queue):
        def _worker(task_queue: Queue, pipe_idx: int):
            while True:
                try:
                    task = task_queue.get_nowait()
                except Empty:
                    break
                self.sampler.execute_one_task(task, pipe_idx=pipe_idx)

        # create threads for each pipeline
        threads = [Thread(target=_worker, args=(task_queue, i)) for i in range(len(self.sampler.pipelines))]

        # start threads
        for thread in threads:
            thread.start()
        # wait for all threads to finish
        for thread in threads:
            thread.join()

    def inference(self):
        log.info(
            f"Starting to execute tasks on {len(self.sampler.pipelines)} GPUs. "
            f"The results will be saved in {self.sampler.output_dir}."
        )
        if len(self.sampler.pipelines) > 1:
            self.prepare_task_queues()

            for i, task_queue in enumerate(self.task_queues):
                log.info(f"Executing tasks (Altenation round: {i + 1}/{len(self.task_queues)}).")
                self.parallel_execute_tasks(task_queue)

            if not check_sampling_results(
                self.sampler.spa_labels, self.sampler.tem_labels, output_dir=self.sampler.output_dir
            ):
                raise ValueError("Sampling failed.")
        else:
            self.sampler.execute_tasks()

    def evaluate(self):
        evaluate_results(
            pred_images_dir=f"{self.sampler.output_dir}/images",
            gt_images_dir=f"{self.sampler.dataset.data_dir}/{self.sampler.dataset.scene_label}/images",
            fmasks_dir=f"{self.sampler.dataset.data_dir}/{self.sampler.dataset.scene_label}/fmasks",
            pred_image_ext=".jpg",
            gt_image_ext=".webp",
            fmask_ext=".png",
            spa_labels=self.sampler.target_spa_labels,
            tem_labels=self.sampler.tem_labels,
            out_metrics_path=f"{self.sampler.output_dir}/metrics.json",
            crop_with_fmask=True,
            background_color="white",
        )

    def to_nerfstudio(self):
        diffuman4d_to_nerfstudio(
            data_dir=f"{self.sampler.dataset.data_dir}/{self.sampler.dataset.scene_label}",
            result_dir=self.sampler.output_dir,
            input_cameras=self.sampler.input_spa_labels,
        )
