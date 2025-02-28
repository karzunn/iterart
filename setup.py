from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name = 'iterart',
    version = '0.3.0',
    author = "Carson O'Ffill",
    author_email = 'offillcarson@gmail.com',
    license = 'MIT',
    description = 'A collection of tools to generate renderings based on iterative functions.',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/karzunn/iterart',
    py_modules = ['iterart'],
    packages = find_packages(),
    install_requires = [requirements],
    python_requires='>=3.12',
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ]
)