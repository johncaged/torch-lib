from setuptools import setup, find_packages
from torch_lib import __version__

README = 'README.md'

setup(
    name='torch_lib',
    version=__version__,
    packages=find_packages(),
    include_package_data=False,
    entry_points={},
    install_requires=[],
    url='https://github.com/johncaged/torch-lib',
    license='GNU General Public License v2.0',
    author='Zikang Liu',
    author_email='573697439@qq.com',
    description='a pytorch lib that helps you to quickly write your training code',
    long_description=open(README, encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
