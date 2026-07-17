import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from typing import Any

from burn_emulator.constants import DOCKERDIR, OUTDIR, Path


def build_image(
    ckpt_path: Path,
    config_paths: Sequence[Path],
    dockerfile: Path,
    tag: str,
    push: bool = False,
):
    dockerpath = DOCKERDIR / dockerfile
    with tempfile.TemporaryDirectory(dir=DOCKERDIR) as staging:
        staging = Path(staging)

        cfg_dest_dir = staging / "configs"
        cfg_dest_dir.mkdir(parents=True, exist_ok=True)
        for cfg in config_paths:
            shutil.copy(cfg, cfg_dest_dir / cfg.name)

        shutil.copy(ckpt_path.parent / "stat.yaml", staging / "stat.yaml")

        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dockerpath),
                "--build-arg",
                f"STAGE_DIR={staging}",
                "--build-arg",
                f"CKPT_NAME={ckpt_path.name}",
                "--build-arg",
                f"CKPT_PATH={str(ckpt_path)}",
                "-t",
                tag,
                str(DOCKERDIR.parent),
            ],
            check=True,
        )

    if push:
        push_image(tag)


def push_image(tag: str):
    try:
        subprocess.run(
            ["docker", "push", tag],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to push image {tag!r} to registry") from e


def package(config_paths: list[Path],
            model_name: str,
            model: dict,
            dockerfile: str,
            artiface_storage: str,
            **kwargs: Any):
    if (ckpt_path := Path(kwargs.get("ckpt_path"))) is None:
        ckpt_dir = OUTDIR / model_name / "checkpoints"
        ckpt_path = sorted(ckpt_dir.glob("*.pt"))[0]
    build_image(
        ckpt_path=ckpt_path, config_paths=config_paths, dockerfile=dockerfile, tag=f"{artifact_storage}/{model_name}"
    )  # TODO: do appropriate versioning
