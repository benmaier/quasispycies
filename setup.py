from setuptools import setup

setup(name='quasispycies',
      version='0.0.1',
      description='Provides classes to compute quasispecies properties on networks.',
      url='https://www.github.com/benmaier/quasispycies',
      author='Benjamin F. Maier',
      author_email='bfmaier@physik.hu-berlin.de',
      license='MIT',
      packages=['quasispycies'],
      install_requires=[
          'numpy',
          'scipy',
          'networkx',
          'networkprops',
      ],
      zip_safe=False)
