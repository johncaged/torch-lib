from setuptools import setup, find_packages

VERSION = '1.0.4'

setup(
    name='torch_train_tool',
    version=VERSION,
    packages=find_packages(),
    include_package_data=False,
    entry_points={},
    install_requires=[],
    url='https://gitee.com/johncaged/torch-tool',
    license='GNU General Public License v2.0',
    author='Zikang Liu',
    author_email='573697439@qq.com',
    description='a pytorch lib that helps you to quickly write your training code'
)
