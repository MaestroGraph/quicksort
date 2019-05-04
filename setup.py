from setuptools import setup

setup(name='dqsort',
      version='0.1',
      description='Differentiable quicksort',
      url='http://github.com/MaestroGraph/quicksort',
      author='Peter Bloem',
      author_email='dqs@peterbloem.nl',
      license='MIT',
      packages=['dqsort'],
      install_requires=[
            'matplotlib',
            'torch',
            'tqdm'
      ],
      zip_safe=False)