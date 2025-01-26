from setuptools import setup, find_packages

setup(
    name='math_function_analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'sympy',
        'matplotlib',
        'ipython'
    ],
    author='Osharov Aleksander',
    author_email='sashosharov@gmail.com',
    description='A library for analyzing mathematical functions',
    url='https://github.com/AlexanderOsharov/math_function_analysis',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)