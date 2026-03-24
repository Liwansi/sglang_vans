import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
from typing import Optional, Union, List, Any, Dict
from types import SimpleNamespace
from loguru import logger
import multiprocessing as mp
import torch

# import sys
#
# sys.path.append("/opt/tiger/sglang")

try:
    from sglang.python import sglang as sgl
except ImportError:
    import sglang as sgl

device_id: str = None
_llm: Optional[sgl.Engine] = None


def _to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(i) for i in x]
    return x


def align_sglang_output_to_vllm_format(outputs, beam_width: int):
    output_beam = [
        outputs[i: i + beam_width]
        for i in range(0, len(outputs), beam_width)
    ]
    vllm_output_format = []
    for o in output_beam:
        classo = _to_ns({"sequences": [{"text": y["text"]} for y in o]})
        vllm_output_format.append(classo)
    return vllm_output_format


class SGLangBatchLLMPool:
    def __init__(
        self,
        device_ids: Optional[List[str]] = None,
        **llm_kwargs
    ):
        self.nproc = len(device_ids)
        self.device_ids = [str(i) for i in range(self.nproc)]
        self.llm_kwargs = llm_kwargs

        logger.info(f"SGLangBatchLLMPool init with device_ids: {self.device_ids}")
        mp.set_start_method('spawn', force=True)
        # Create a queue to pass device IDs to workers
        self.queue = Queue()
        for device_id in self.device_ids:
            self.queue.put(device_id)

        # Initialize the process pool
        self.pool = ProcessPoolExecutor(
            max_workers=self.nproc,
            initializer=self._worker_init,
            initargs=(self.llm_kwargs, self.queue)
        )

        # Sample prompts
        prompts = [
            {"prompt": "force trigger all devices"},
        ]
        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 3,
            "n": 2,
        }
        futures = [self.beam_search(prompts, sampling_params) for x in range(self.nproc)]
        for future in as_completed(futures):
            results = future.result()
            for result in results:
                logger.info(result)
        logger.info(f"SGLangBatchLLMPool initialized with {self.nproc} processes with device_ids: {self.device_ids}")

    def worker_num(self):
        return self.nproc

    @staticmethod
    def _worker_init(llm_kwargs: dict, queue: Queue):
        """
        Initialize a worker process with an SGLang Engine instance.

        Args:
            llm_kwargs: Additional arguments for SGLang Engine
            queue: Queue to get device ID from
        """
        # Get device ID from queue
        global device_id
        device_id = queue.get()

        # Set environment variable for NPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        # os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id
        logger.info(f"Worker[{os.getpid()}] on device {device_id} initialized")
        if hasattr(torch, "npu"):
            llm_kwargs["base_gpu_id"] = int(device_id)
        # Create the SGLang Engine instance
        global _llm
        _llm = sgl.Engine(**llm_kwargs)
        logger.info(f"Worker[{os.getpid()}] on device {device_id} initialized with _llm instance at {hex(id(_llm))}")

    def beam_search(
        self,
        prompts: List[str],
        params: Dict,
    ):
        # check 以下参数如何和主进程通信
        future = self.pool.submit(
            self._beam_search_worker,
            prompts,
            params
        )
        return future

    @staticmethod
    def _beam_search_worker(
        prompts: List[str],
        params: Dict
    ):
        """
        Worker function for beam_search
        """
        global _llm, device_id
        output = _llm.generate([i["prompt"] for i in prompts], params)
        return align_sglang_output_to_vllm_format(output, params["n"])

    def shutdown(self, wait: bool = True):
        """
        Shutdown the process pool.

        Args:
            wait: Whether to wait for all processes to complete
        """
        self.pool.shutdown(wait=wait)
        logger.info("SGLangBatchLLMPool shutdown complete")


# Example usage
def example():
    # Sample prompts
    prompts = [
        {"prompt": "Hello, my name is"},
        {"prompt": "The president of the United States is"},
        {"prompt": "The capital of France is"},
        {"prompt": "The future of AI is"},
        {"prompt": "Python is a popular programming language because"},
        {"prompt": "The largest planet in our solar system is"},
        {"prompt": "Quantum computing could revolutionize"},
        {"prompt": "Climate change is a pressing issue because"},
    ]

    # Create sampling params
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 3,
        "n": 100,
    }
    engine_args = {
        "model_path": "/home/weights/Qwen3-8B",
        "enable_beam_search": True,
        "disable_overlap_schedule": True,
        "chunked_prefill_size": -1,
        "mem_fraction_static": 0.8,
        # "attention_backend": "ascend",
    }
    device_ids = [str(x) for x in range(4)]
    # Initialize pool with 1 process using device 0
    pool = SGLangBatchLLMPool(
        device_ids=device_ids,
        **engine_args,
        # Additional LLM kwargs can be passed here
    )

    try:
        # Generate using the pool
        logger.info("Generating responses...")
        start_time = time.time()

        # future_to_index = {pool.beam_search(prompts, sampling_params): x for x in range(len(device_ids))}
        # while True:
        #     for future in as_completed(future_to_index):
        #         index = future_to_index.pop(future)
        #         f = pool.beam_search(prompts, sampling_params)
        #         future_to_index[f] = index
        #     # logger.info(f"Generate {prompts} outputs {outputs}")

        batch_size = len(prompts) // len(device_ids)  # 8//4=2
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            future = pool.beam_search(batch, sampling_params)
            outputs = future.result()
            logger.info(f"Generate {prompts} outputs {outputs}")

        elapsed = time.time() - start_time
        logger.info(f"Generation completed in {elapsed:.2f} seconds")
        pool.shutdown()

    finally:
        # Shutdown the pool
        pool.shutdown()


if __name__ == "__main__":
    example()
