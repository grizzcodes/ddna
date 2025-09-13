from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ddna",
    version="0.1.0",
    author="GrizzCodes",
    author_email="",
    description="Screenplay DNA System - Visual consistency through scene DNA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/grizzcodes/ddna",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "pydantic>=2.5.0",
        "pillow>=10.1.0",
        "numpy>=1.24.3",
        "click>=8.1.7",
        "tqdm>=4.66.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ddna=ddna.cli:main",
        ],
    },
)