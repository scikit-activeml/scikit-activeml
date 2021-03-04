import setuptools


def readme():
    with open('README.rst', 'r') as f:
        return f.read()


setuptools.setup(name='scikit-activeml',
                 version='0.0.1',
                 description='The package scikit-activeml is a library of '
                             'that covers the most relevant query strategies '
                             'in active learning and implements tools to work '
                             'with partially labeled data.',
                 long_description=readme(),
                 long_description_content_type='text/x-rst',
                 classifiers=[
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: BSD License',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Operating System :: OS Independent',
                 ],
                 keywords=['active learning', 'machine learning',
                           'semi-supervised learning', 'data mining',
                           'pattern recognition', 'artificial intelligence'],
                 url='https://scikit-activeml.readthedocs.io/en/latest/',
                 author='Daniel Kottke',
                 author_email='daniel.kottke@uni-kassel.de',
                 license='BSD 3-Clause License',
                 packages=setuptools.find_packages(),
                 install_requires=[
                     'joblib~=1.0.1',
                     'numpy~=1.20.1',
                     'scipy~=1.6.1',
                     'scikit-learn~=0.24.1',
                     'matplotlib~=3.3.4',
                     'iteration-utilities~=0.11.0'
                 ],
                 test_suite='nose.collector',
                 tests_require=['nose'],
                 include_package_data=True,
                 zip_safe=False)
