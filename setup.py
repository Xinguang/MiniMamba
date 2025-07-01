from setuptools import setup, find_packages
import os

# Get version from _version.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'minimamba', '_version.py')
    with open(version_file, 'r') as f:
        exec(f.read())
    return locals()['__version__']

# Read the long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Production-ready PyTorch implementation of Mamba with optimized parallel scan, modular architecture, and comprehensive caching support."

setup(
    name="minimamba",
    version=get_version(),
    description="Production-ready PyTorch implementation of Mamba (Selective State Space Model) with optimized parallel scan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Xinguang",
    author_email="minimanba.github@kansea.com",
    url="https://github.com/Xinguang/MiniMamba",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "examples": [
            "matplotlib>=3.0",
            "pandas>=1.2.0",
            "jupyter>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="mamba, state-space-model, transformer, attention, deep-learning, pytorch, nlp, time-series",
    python_requires=">=3.8",
    project_urls={
        "Bug Reports": "https://github.com/Xinguang/MiniMamba/issues",
        "Source": "https://github.com/Xinguang/MiniMamba",
        "Documentation": "https://github.com/Xinguang/MiniMamba/blob/main/README.md",
        "Improvements": "https://github.com/Xinguang/MiniMamba/blob/main/IMPROVEMENTS.md",
    },
    # entry_points removed - not needed for this package
    include_package_data=True,
    zip_safe=False,
)
