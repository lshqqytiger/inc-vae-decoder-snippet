import os
import json
import shutil
import importlib
from typing import Dict, Any, List, Union, Tuple, Type
import diffusers
import onnxruntime as ort
from olive.workflows import run
from olive.model import ONNXModel
from olive_script import config


model_name = "runwayml/stable-diffusion-v1-5"
model_id = "--".join(model_name.split("/"))
submodels_sd = ("vae_decoder",) # ("text_encoder", "unet", "vae_encoder", "vae_decoder",)


def load_init_dict(cls: Type[diffusers.DiffusionPipeline], path: os.PathLike):
    merged: Dict[str, Any] = {}
    extracted = cls.extract_init_dict(diffusers.DiffusionPipeline.load_config(path))
    for dict in extracted:
        merged.update(dict)
    merged = merged.items()
    R: Dict[str, Tuple[str]] = {}
    for k, v in merged:
        if isinstance(v, list):
            if k not in cls.__init__.__annotations__:
                continue
            R[k] = v
    return R


def load_submodel(path: os.PathLike, is_sdxl: bool, submodel_name: str, item: List[Union[str, None]], **kwargs_ort):
    lib, atr = item
    if lib is None or atr is None:
        return None
    library = importlib.import_module(lib)
    attribute = getattr(library, atr)
    path = os.path.join(path, submodel_name)
    if issubclass(attribute, diffusers.OnnxRuntimeModel):
        return diffusers.OnnxRuntimeModel.load_model(
            os.path.join(path, "model.onnx"),
            **kwargs_ort,
        ) if is_sdxl else diffusers.OnnxRuntimeModel.from_pretrained(
            path,
            **kwargs_ort,
        )
    return attribute.from_pretrained(path)


def load_submodels(path: os.PathLike, is_sdxl: bool, init_dict: Dict[str, Type], **kwargs_ort):
    loaded = {}
    for k, v in init_dict.items():
        if not isinstance(v, list):
            loaded[k] = v
            continue
        try:
            loaded[k] = load_submodel(path, is_sdxl, k, v, **kwargs_ort)
        except Exception:
            pass
    return loaded


def run_olive():
    ort.set_default_logger_severity(4)

    dir_name = f"models--{model_id}"
    snapshots_dir = os.path.join("./model_cache", dir_name, "snapshots")
    snapshots = os.listdir(snapshots_dir)
    in_dir = os.path.join(snapshots_dir, snapshots[0])
    out_dir = os.path.join("./model_quantized", dir_name)

    init_dict = load_init_dict(diffusers.StableDiffusionPipeline, in_dir)

    try:
        shutil.rmtree("cache", ignore_errors=True)
        shutil.rmtree("footprints", ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)

        shutil.copytree(
            in_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
        )

        quantized_model_paths = {}

        for submodel in submodels_sd:
            print(f"\nProcessing {submodel}")

            with open(os.path.join("./olive", f"{submodel}.json"), "r") as config_file:
                olive_config = json.load(config_file)
            olive_config["input_model"]["config"]["model_path"] = in_dir

            run(olive_config)

            with open(os.path.join("footprints", f"{submodel}_gpu-dml_footprints.json"), "r") as footprint_file:
                footprints = json.load(footprint_file)
            processor_final_pass_footprint = None
            for _, footprint in footprints.items():
                if footprint["from_pass"] == olive_config["passes"][olive_config["pass_flows"][-1][-1]]["type"]:
                    processor_final_pass_footprint = footprint

            assert processor_final_pass_footprint, "Failed to quantize model"

            quantized_model_paths[submodel] = ONNXModel(
                **processor_final_pass_footprint["model_config"]["config"]
            ).model_path

            print(f"Processed {submodel}")

        for submodel in submodels_sd:
            src_path = quantized_model_paths[submodel]
            src_parent = os.path.dirname(src_path)
            dst_parent = os.path.join(out_dir, submodel)
            dst_path = os.path.join(dst_parent, "model.onnx")
            if not os.path.isdir(dst_parent):
                os.mkdir(dst_parent)
            shutil.copyfile(src_path, dst_path)

            data_src_path = os.path.join(src_parent, (os.path.basename(src_path) + ".data"))
            if os.path.isfile(data_src_path):
                data_dst_path = os.path.join(dst_parent, (os.path.basename(dst_path) + ".data"))
                shutil.copyfile(data_src_path, data_dst_path)

            weights_src_path = os.path.join(src_parent, "weights.pb")
            if os.path.isfile(weights_src_path):
                weights_dst_path = os.path.join(dst_parent, "weights.pb")
                shutil.copyfile(weights_src_path, weights_dst_path)
        del quantized_model_paths

        kwargs = {}

        for submodel in submodels_sd:
            kwargs[submodel] = diffusers.OnnxRuntimeModel.from_pretrained(
                os.path.join(out_dir, submodel),
                provider="DmlExecutionProvider",
            )
            if submodel in init_dict:
                del init_dict[submodel] # already loaded as OnnxRuntimeModel.
        kwargs.update(load_submodels(in_dir, False, init_dict)) # load others.
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

        pipeline = diffusers.OnnxStableDiffusionPipeline(**kwargs)
        model_index = json.loads(pipeline.to_json_string())
        del pipeline

        for k, v in init_dict.items(): # copy missing submodels. (ORTStableDiffusionXLPipeline)
            if k not in model_index:
                model_index[k] = v

        with open(os.path.join(out_dir, "model_index.json"), 'w') as file:
            json.dump(model_index, file)

        return out_dir
    except Exception:
        import traceback
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    pipeline = diffusers.DiffusionPipeline.from_pretrained(model_name, cache_dir="./model_cache")
    del pipeline

    config.is_sdxl = False

    config.width = 512
    config.height = 512

    config.batch_size = 1

    if config.is_sdxl:
        config.cross_attention_dim = 2048
        config.time_ids_size = 6
    else:
        config.cross_attention_dim = 256 + config.height
        config.time_ids_size = 5

    print(run_olive())
