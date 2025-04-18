"""Setup script for RecSys-Lite.

This is used for installing the package in development mode in Docker.
"""

from setuptools import setup, find_namespace_packages

if __name__ == "__main__":
    setup(
        name="recsys-lite",
        version="0.1.0",
        description="Lightweight recommendation system for small e-commerce shops",
        author="RecSys-Lite Team",
        author_email="info@recsys-lite.com",
        packages=find_namespace_packages(where="src"),
        package_dir={"": "src"},
        entry_points={
            "console_scripts": [
                "recsys-lite=recsys_lite.cli:app",
            ],
        },
        python_requires=">=3.11",
        # Dependencies are installed from requirements.txt in Docker
    )
