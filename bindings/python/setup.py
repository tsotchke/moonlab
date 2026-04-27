"""Setup script for Moonlab Python bindings"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
repo_root = Path(__file__).parent.parent.parent
readme_path = repo_root / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Single source of truth for the version is VERSION.txt at the repo root,
# matching CMakeLists.txt + pyproject.toml + bindings/rust/moonlab/Cargo.toml.
# Convert PEP-440-incompatible "X.Y.Z-dev" -> "X.Y.Z.dev0" so pip is happy.
version_path = repo_root / "VERSION.txt"
_raw_version = version_path.read_text().strip() if version_path.exists() else "0.0.0"
if _raw_version.endswith("-dev"):
    _version = _raw_version[: -len("-dev")] + ".dev0"
else:
    _version = _raw_version.replace("-", ".")

setup(
    name="moonlab",
    version=_version,
    author="tsotchke",
    description="High-performance quantum computing simulator for Apple Silicon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsotchke/moonlab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "ml": ["torch>=1.9.0", "tensorflow>=2.5.0"],
        "viz": ["matplotlib>=3.3.0", "plotly>=5.0.0"],
        "dev": ["pytest>=6.0.0", "black", "flake8", "mypy"],
    },
)