import setuptools
import os

requirementPath = 'requirements.txt'
reqs = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        reqs = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CSI2Depth",
    version="0.0.1",
    author="Constantino Álvarez Casado, Manuel Lage Cañellas, Janne Mustaniemi, Matteo Pedone, Olli Silvén, Miguel Bordallo López",
    author_email="constantino.alvarezcasado@oulu.fi",
    description="Package to estimate depht images from WiFi CSI data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arritmic/csi2depth",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=reqs
)