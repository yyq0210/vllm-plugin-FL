# Copyright (c) 2026 BAAI. All rights reserved.
"""Select the bounded Iluvatar scheduler without importing the engine early."""

SCHEDULER_PATH = (
    "vllm_fl.worker.kv_reservation_scheduler.KVReservationAsyncScheduler"
)


def configure_kv_reservation(vllm_config, vendor_name: str) -> bool:
    scheduler = vllm_config.scheduler_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    if (
        vendor_name != "iluvatar"
        or model is None
        or scheduler.scheduler_cls is not None
        or scheduler.async_scheduling is not True
        or not scheduler.scheduler_reserve_full_isl
        or model.is_encoder_decoder
        or model.is_multimodal_model
        or model.runner_type != "generate"
        or vllm_config.speculative_config is not None
        or vllm_config.kv_transfer_config is not None
        or vllm_config.ec_transfer_config is not None
        or any(
            getattr(parallel, name, None) != 1
            for name in (
                "tensor_parallel_size", "pipeline_parallel_size",
                "data_parallel_size", "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        )
    ):
        return False
    scheduler.scheduler_cls = SCHEDULER_PATH
    return True
