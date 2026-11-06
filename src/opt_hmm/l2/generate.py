"""L2 sources with vLLM (Linux/CUDA): per prompt a watermarked W and/or an unwatermarked U generation."""
from ..io import read_json, write_json
from ..l0.generate import generate as generate_sources
from ..paths import project_path
from .config import validate_l2
from .plan import plan_jobs
from .stages import default_progress_path, install_overall


def generate(config, model, prompts_dir, output, side, tensor_parallel=1, gpu_memory=.8, dtype="float16",
             progress_path=None):
    validate_l2(config)
    spec = config["l2"]["sides"][side]
    roles = set(spec["roles"])
    stage = "G1" if side == "train" else "G2"
    install_overall(config, progress_path or default_progress_path(output), (stage,))

    def planner(prompts, cfg):
        return plan_jobs([d for d in prompts if d["role"] in roles], cfg)

    result = generate_sources(config, model, prompts_dir, output, tensor_parallel, gpu_memory, dtype,
                              validate=validate_l2, planner=planner, key=spec["key"], key_id=spec["key_id"],
                              chunk=config["l2"].get("generate_chunk", 200),
                              tick_stage=stage)
    path = project_path(output) / "metadata.json"
    write_json(path, {**read_json(path), "side": side})
    return result
