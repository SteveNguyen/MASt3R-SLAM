from setuptools import setup

# `curope` and `asmk` are local sibling packages. Upstream embedded them as
# `pkg @ file:///abs/path` in install_requires, which uv bakes into uv.lock as
# absolute paths and breaks portability across machines. The root project's
# pyproject.toml already declares them via [tool.uv.sources] / lists them as
# direct deps, so we just name them here without the file:// URI.
setup(
    install_requires=[
        "scikit-learn",
        "roma",
        "gradio",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "scipy",
        "einops",
        "trimesh",
        "tensorboard",
        "pyglet",
        "huggingface-hub[torch]>=0.22",
        "curope",
        "asmk",
    ],
)
