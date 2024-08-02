from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(name='csr_detector',
          version='1.0',
          author='Ali Tourani',
          description='CSR detector methodology using image registration',
          url='https://github.com/snt-arg/csr_detector',
          packages=find_packages(
              include=['vision']),
          install_requires=[
              'numpy',
              'opencv-python',
          ],
          )
