"""Setup script for RecSys-Lite.

This is only used to help with installing dependencies that have issues with Poetry and PEP 517.
"""

from setuptools import setup

if __name__ == "__main__":
    setup(
        name="recsys-lite",
        version="0.1.0",
        description="Lightweight recommendation system for small e-commerce shops",
        author="RecSys-Lite Team",
        author_email="info@recsys-lite.com",
        packages=["recsys_lite"],
        package_dir={"": "src"},
        install_requires=[
            "duckdb>=0.8.1",
            "typer>=0.8.0",
            "implicit>=0.7.0",
            "gensim>=4.3.1",
            # LightFM is installed separately
            # "lightfm==1.16",
            "optuna>=3.3.0",
            "faiss-cpu>=1.7.4",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
        ],
    )