from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='imarker_algorithms',
        version='1.0.0',
        author='Ali Tourani',
        author_email='ali.tourani@uni.lu',
        description='Algorithms to detect iMarkers',
        long_description='A Python package providing computer vision algorithms used in iMarker detection.',
        long_description_content_type='text/plain',
        url='https://github.com/snt-arg/iMarker_algorithms',
        packages=find_packages(include=['vision', 'vision.*']),
        install_requires=[
            'numpy>=1.24.4',
            'opencv-python>=4.10.0.84'
        ],
        python_requires='>=3.8',
    )
