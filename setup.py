"""Setup script for spine-seg-cte package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="spine-seg-cte",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Complete bone analysis pipeline for vertebral CT segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spine-seg-cte",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
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
        "pandas>=1.3.0",
        "nibabel>=3.2.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.60.0",
        "TotalSegmentator>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "spine-seg-cte=spine_seg_cte.cli:main",
        ],
    },
)
