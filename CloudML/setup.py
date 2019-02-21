from setuptools import setup, find_packages

#Setup parameters for Google Cloud ML Engine
setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Run keras on gcloud ml-engine',
      author='Luis',
      author_email='lpedroj3@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py,'
      ],
      zip_safe=False)