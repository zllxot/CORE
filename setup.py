from setuptools import find_packages, setup

setup(
    name='CORE',
    version='1.1.0',
    description='Cooperative reconstruction for multi-agent perception',
    packages=find_packages(include=['opencood', 'opencood.*']),
    package_data={
        'opencood.hypes_yaml': ['*.yaml'],
        'opencood.utils': ['*.pyx'],
    },
    python_requires='>=3.7',
)
