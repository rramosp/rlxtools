from setuptools import setup
exec(open('rlxtools/version.py').read())

setup(name='rlxtools',
      version=__version__,
      description='rlx tools',
      url='http://github.com/rramosp/rlxtools',
      install_requires=['matplotlib','numpy', 'pykalman', 'pandas', 'deepdish'],
      author='rlx',
      author_email='rulix.rp@gmail.com',
      license='MIT',
      packages=['rlxtools'],
      include_package_data=True,
      zip_safe=False)
