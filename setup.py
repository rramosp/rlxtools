from setuptools import setup


setup(name='rlxtools',
      version=open('rlxtools/__version__').read().rstrip(),
      description='rlx tools',
      url='http://github.com/rramosp/rlxtools',
      author='rlx',
      author_email='rulix.rp@gmail.com',
      license='MIT',
      packages=['rlxtools'],
      zip_safe=False)
