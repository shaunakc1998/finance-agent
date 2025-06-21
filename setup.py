from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="finance-agent",
    version="0.1.0",
    author="Finance Agent Team",
    author_email="example@example.com",
    description="A financial analysis and forecasting tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/finance-agent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt", "r")
        if not line.startswith("#") and line.strip()
    ],
    include_package_data=True,
)
