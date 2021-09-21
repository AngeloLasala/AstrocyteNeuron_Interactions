from setuptools import setup, find_packages

setup(
    name='AstrocyteNeuron_Interactions',
    version='0.0.1',
    description='Packages for easy computation of astrocyte-neuron interaction',
    url='https://github.com/AngeloLasala/Astrocyte-Neuron_Interactions',
    author='Lasala Angelo',
    author_email='lasala.angelo@gmail.com',
    license='gnu general public license',
    packages = find_packages(),
    install_requires=['numpy', 'requests', 'scikit-learn'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)