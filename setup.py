from setuptools import setup, find_packages

setup(
    name='grace',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Author Name',
    author_email='vignif@gmail.com',
    description='Description of my package',
    packages=find_packages(),
    package_dir = {"": "."},
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1', 'sympy >= 1.1.1'],
)