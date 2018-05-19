from distutils.core import setup
import imp
import os

from setuptools import find_packages

setup_requires = []
install_requires = [
    'chainer>=2.0',
    'chainer-chemistry>=0.3.0',
]


here = os.path.abspath(os.path.dirname(__file__))
__version__ = imp.load_source(
    '_version', os.path.join(here,
                             'chainer_pointnet', '_version.py')).__version__

setup(name='chainer-pointnet',
      version=__version__,
      description='Chainer PointNet',
      author='corochann',
      author_email='',
      packages=find_packages(),
      license='MIT',
      # url='',
      setup_requires=setup_requires,
      install_requires=install_requires
      )
