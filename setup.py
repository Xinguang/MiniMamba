from setuptools import setup, find_packages

setup(
    name="mamba",
    version="0.1.0",
    description="A clean and efficient PyTorch implementation of the Mamba architecture (Selective State Space Models)",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/mamba",  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
