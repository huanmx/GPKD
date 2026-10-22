# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from pathlib import Path
import re
from typing import List, Tuple

from setuptools import setup, find_namespace_packages


NAME = "gpkd"
DESCRIPTION = "Generation-to-Perception Knowledge Distillation built on DINOv2 and REG."

URL = "https://github.com/huanmx/GPKD"
AUTHOR = "GPKD contributors"
REQUIRES_PYTHON = ">=3.10.0,<3.12"
HERE = Path(__file__).parent


try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


def get_requirements(path: str = HERE / "requirements.txt") -> Tuple[List[str], List[str]]:
    requirements = []
    extra_indices = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("--extra-index-url "):
                extra_indices.append(line[18:])
                continue
            requirements.append(line)
    return requirements, extra_indices


def get_package_version() -> str:
    with open(HERE / "dinov2/__init__.py") as f:
        result = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if result:
            return result.group(1)
    raise RuntimeError("Can't get package version")


requirements, extra_indices = get_requirements()
version = "0.1.0"
gpu_requirements, _ = get_requirements(HERE / "requirements-gpu.txt")
dev_requirements, _ = get_requirements(HERE / "requirements-dev.txt")
extras_requirements, _ = get_requirements(HERE / "requirements-extras.txt")


setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_namespace_packages(include=["dinov2", "dinov2.*"]),
    package_data={
        "": ["*.yaml"],
    },
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "extras": extras_requirements,
        "gpu": [line for line in gpu_requirements if line and not line.startswith("#")],
    },
    dependency_links=extra_indices,
    include_package_data=True,
    license="Apache-2.0; bundled components retain their own licenses",
    license_files=("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "LICENSE_CELL_DINO_CODE", "LICENSE_CELL_DINO_MODELS", "LICENSE_XRAY_DINO_MODEL",
                   "dinov2/gpkd/generation/LICENSE", "dinov2/thirdparty/CLIP/LICENSE"),
    classifiers=[
        # Trove classifiers: https://github.com/pypa/trove-classifiers/blob/main/src/trove_classifiers/__init__.py
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
