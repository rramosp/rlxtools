from setuptools import setup
exec(open('rlxtools/__version__').read())


setup(name='rlxtools',
      version=__version__,
      description='rlx tools',
      url='http://github.com/rramosp/rlxtools',
      author='rlx',
      author_email='rulix.rp@gmail.com',
      license='MIT',
      packages=['rlxtools'],
      zip_safe=False)
