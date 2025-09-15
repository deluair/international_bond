"""
Setup configuration for International Bond Relative Value System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="international-bond-system",
    version="1.0.0",
    author="International Bond System Team",
    author_email="contact@internationalbond.com",
    description="A comprehensive system for international bond relative value analysis and trading",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/international-bond/bond-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
            "bandit>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.1.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=7.0.0",
        ],
        "ml": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
        ],
        "web": [
            "streamlit>=1.25.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "international-bond=main:main",
            "bond-system=main:main",
            "bond-analyzer=main:analyze_mode",
            "bond-trader=main:trading_mode",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv", "*.txt"],
    },
    zip_safe=False,
    keywords="finance, bonds, fixed-income, trading, risk-management, portfolio-optimization",
    project_urls={
        "Bug Reports": "https://github.com/international-bond/bond-system/issues",
        "Source": "https://github.com/international-bond/bond-system",
        "Documentation": "https://international-bond-system.readthedocs.io/",
    },
)