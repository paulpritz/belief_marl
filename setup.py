from setuptools import setup, find_packages

setup(
    name="marl",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "gym",
        "scipy",
        "numpy",
        "pyyaml",
        "matplotlib",
        "gymnasium",
    ],
    scripts=[
        "./marl/bin/run_training_belief",
    ],
)
