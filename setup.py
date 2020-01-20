from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

dependencies = [
        'tensorflow>2.0',
        'numpy>=1.16'
]

setup(
    name='vtmm',
    version='0.1',
    description='Vectorized transfer matrix method (TMM) for computing the optical reflection and transmission of multilayer planar stacks',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/fancompute/vtmm',
    author='Ian Williamson',
    author_email='ian.williamson@ieee.org',
    license='MIT',
    packages=find_packages(),
    install_requires=dependencies,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
