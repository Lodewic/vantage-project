from setuptools import find_packages, setup

setup(
    name='vantage-project',
    packages=find_packages(),
    version='0.0.1',
    description='A data science case for Vantage AI',
    author='Lodewic van Twillert',
    license='Apache License 2.0',
    long_description="README.md",

    python_requires='>3.5',

    install_requires=[
                     "click",
                     "python-dotenv>=0.5.1",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
