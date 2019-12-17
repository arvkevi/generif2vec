from setuptools import setup, find_packages
from codecs import open as copen
from os import path

from generif2vec import __project__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with copen(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the dependencies and installs
with copen(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]

version = {}
with open("generif2vec/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="generif2vec",
    entry_points={"console_scripts": ["{} = {}.__main__:main".format(__project__, __project__)]},
    version=version["__version__"],
    description="Tools to work with Doc2Vec and NCBIs Gene Reference Into Function (RIF).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arvkevi/generif2vec",
    download_url="https://github.com/arvkevi/generif2vec/tarball/"
    + version["__version__"],
    license="BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    keywords="doc2vec genetics genomics",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author="Kevin Arvai",
    install_requires=install_requires,
    author_email="arvkevi@gmail.com",
)
