from setuptools import setup, find_packages


install_requires = []
package_data = {}

setup(
    name='IRNet',
    version='0.0.1',
    py_modules=['IRNet'],
    # See: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    install_requires=install_requires,
    package_data=package_data,
)
