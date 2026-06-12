import importlib.util
import os
import sys
import time
from pathlib import Path

from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


_EXTENSION = None


def _sources_are_older_than(path: Path, sources) -> bool:
    try:
        output_mtime = path.stat().st_mtime
    except OSError:
        return False
    for source in sources:
        try:
            if source.stat().st_mtime > output_mtime:
                return False
        except OSError:
            return False
    return True


def _load_prebuilt_extension(path: Path):
    module_name = "ring_allreduce_native"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load extension spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _clear_stale_build_lock(build_dir: Path) -> None:
    """Avoid indefinite waits when torch extension loading leaves build/lock behind."""
    rank = os.environ.get("RANK")
    if rank is not None and rank != "0":
        return

    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return

    stale_after = float(os.environ.get("RING_EXT_STALE_LOCK_SECONDS", "60"))
    if stale_after < 0:
        return

    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return

    if age < stale_after:
        return

    try:
        lock_path.unlink()
        print(
            f"Removed stale ring_allreduce_ext build lock "
            f"({lock_path}, age={age:.1f}s)",
            flush=True,
        )
    except FileNotFoundError:
        return


def load_ring_allreduce_ext(verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    cuda_home = os.environ.get("CUDA_HOME", "/share/apps/cuda-11.8")
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ.setdefault("CC", "/share/apps/gcc-9.2.0/bin/gcc")
    os.environ.setdefault("CXX", "/share/apps/gcc-9.2.0/bin/g++")
    os.environ.setdefault("CUDAHOSTCXX", "/share/apps/gcc-9.2.0/bin/g++")
    if cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = cuda_home

    here = Path(__file__).resolve().parent
    eden_root = Path(
        os.environ.get(
            "EDEN_UTILS_ROOT",
            "/cluster/project2/gcreduce_data/dynamiq_artifact/cuda_kernels/src/srrcomp/eden_utils",
        )
    )
    eden_csrc = eden_root / "csrc"
    build_dir = here / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        here / "ring_allreduce_bindings.cpp",
        here / "ring_allreduce_backend.cu",
        eden_csrc / "cuda_hadamard.cu",
        eden_csrc / "cuda_hierarchical_mee.cu",
        eden_csrc / "cuda_packing.cu",
    ]
    _clear_stale_build_lock(build_dir)

    prebuilt = build_dir / "ring_allreduce_native.so"
    if (
        os.environ.get("RING_EXT_FORCE_JIT", "").lower() not in ("1", "true", "yes", "on")
        and _sources_are_older_than(prebuilt, sources)
    ):
        _EXTENSION = _load_prebuilt_extension(prebuilt)
        return _EXTENSION

    _EXTENSION = load(
        name="ring_allreduce_native",
        sources=[str(source) for source in sources],
        extra_include_paths=[str(here), str(eden_csrc)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-ccbin=/share/apps/gcc-9.2.0/bin/g++",
        ],
        extra_ldflags=["-lrdmacm", "-libverbs", "-lcurand"],
        build_directory=str(build_dir),
        verbose=verbose,
    )
    return _EXTENSION
