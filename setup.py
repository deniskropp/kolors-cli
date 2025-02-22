from setuptools import setup, find_packages

setup(
    name='kolors-cli',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'accelerate==0.27.2',
        'diffusers>=0.29.0',
        'invisible_watermark==0.2.0',
        'torch==2.2.0',
        'transformers==4.42.4',
        'sentencepiece==0.1.99',
        'Pillow'
    ],
    entry_points={
        'console_scripts': [
            'kolors-cli=kolors_cli.kolors_cli:main',
        ],
    },
    author='Denis Kropp',
    author_email='dok@directfb1.org',
    description='A command-line interface for generating images using the Kolors model.',
    url='https://github.com/deniskropp/kolors-cli',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
