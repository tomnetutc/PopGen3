from setuptools import setup, find_packages

setup(
    name="popgen3",
    version='3.0.4',
    author="Fan Yu",
    author_email="fanyu4@asu.edu",
    description="A population synthesis and sample weighting tool for transportation planning.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chnfanyu/PopGen3",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.21.6",
        "pandas>=1.3.0",
        "scipy>=1.10.1",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "popgen=popgen.cli:main"
        ]
    },
    include_package_data=True,
)
