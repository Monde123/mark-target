from setuptools import setup, find_packages

setup(
    name="isolate-marker-retargeting",
    version="1.0.0",
    description="Universal 3D Surface Marker & Multi-Point Kabsch Retargeting Framework",
    author="Pipeline Signes Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "gltf": ["pygltflib>=1.15.0"],
        "all": ["pygltflib>=1.15.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
