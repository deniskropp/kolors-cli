from setuptools import setup, find_packages

setup(
    name='kolors',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'accelerate==0.27.2',
        'diffusers==0.28.2',
        'invisible_watermark==0.2.0',
        'torch==2.2.0',
        'transformers==4.42.4',
        'sentencepiece==0.1.99',
    ],
    entry_points={
        'console_scripts': [
            'kolors=kolors.app:Kolors.launch'
        ],
    },
)
