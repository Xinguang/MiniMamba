from setuptools import setup, find_packages

setup(
    name="minimamba",
    version="0.2.0",
    description="Minimal PyTorch implementation of Mamba (Selective State Space Model)",
    author="Xinguang",
    author_email="minimanba.github@kansea.com",
    url="https://github.com/Xinguang/MiniMamba",
    packages=find_packages(),
    install_requires=["torch>=1.12.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
