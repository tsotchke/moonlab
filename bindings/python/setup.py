"""Setup script for Moonlab Python bindings"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent.parent.parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="moonlab",
    version="0.1.1",
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