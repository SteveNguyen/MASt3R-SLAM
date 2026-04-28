from setuptools import setup

# `imgui` (pyimgui) is a local sibling package. Upstream embedded it as
# `pkg @ file:///abs/path` in install_requires, which uv bakes into uv.lock
# as an absolute path and breaks portability across machines. The root
# project's pyproject.toml already declares imgui via [tool.uv.sources] /
# lists it as a direct dep, so we just name it here without the file:// URI.
setup(
    install_requires=[
        "imgui",
        "moderngl==5.12.0",
        "moderngl-window==2.4.6",
        "glfw",
        "pyglm",
        "msgpack",
        "numpy",
        "matplotlib",
        "trimesh[easy]",
    ]
)
