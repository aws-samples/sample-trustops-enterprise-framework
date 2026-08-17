"""Setup configuration for TrustOps CLI."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trustops",
    version="0.1.0",
    author="TrustOps Team",
    description="Trust-first fine-tuning and evaluation for GenAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "boto3>=1.34.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
        "numpy>=1.24.0",
        "pydantic>=2.5.0",
    ],
    entry_points={
        "console_scripts": [
            "trustops=cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
