import shutil
import subprocess
import tempfile

from typing import Any, Sequence

from burn_emulator.constants import Path, DOCKERDIR


def build_image(
    ckpt_path: Path,
    config_paths: Sequence[Path],
    dockerfile: Path,
    tag: str,
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
                "docker", "build",
                "-f", str(dockerpath),
                "--build-arg", f"STAGE_DIR={staging}",
                "--build-arg", f"CKPT_NAME={ckpt_path.name}"
                "--build-arg", f"CKPT_PATH={str(ckpt_path)}"
                "-t", tag,
                DOCKERDIR.parent
            ],
            check=True,
        )

        # TODO: push to artifact store
  
def package(config_paths,
            model_name: str,
            model: dict,
            dockerfile: str,
            **kwargs: Any):
    model_name = kwargs.get("model_name")
    model_spec = kwargs.get("model").get("class_path")

    if (ckpt_path := Path(kwargs.get("ckpt_path"))) is None:
        ckpt_dir = OUTDIR / model_name / "checkpoints"
        ckpt_path = sorted(ckpt_dir.glob("*.pt"))[0]
    build_image(ckpt_path=ckpt_path, 
                config_paths=config_paths,
                dockerfile=dockerfile,
                tag=model_name) # TODO: do appropriate versioning
