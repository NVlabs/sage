from setuptools import setup, find_packages

setup(
    name="m2t2",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        line for line in open('requirements.txt').readlines()
        if "@" not in line
    ],
    description="Multi-Task Masked Transformer",
    author="Wentao Yuan",
    author_email="wentaoy@nvidia.com",
    license="MIT Software License",
    url="https://m2-t2.github.io",
    keywords="robotics manipulation learning computer-vision",
    classifiers=[
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
)
