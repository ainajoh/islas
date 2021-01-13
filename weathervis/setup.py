from setuptools import setup

setup(
    name='weathervis',
    version='1.0',
    packages=['weathervis', 'weathervis.plots', 'weathervis.plots.basemap', 'weathervis.plots.cartopy'],
    url='',
    license='',
    author='ainajoh',
    author_email='aina.johannessen@uib.no',
    description='',
    python_requires='>=3.6, <4',
    install_requires=[
    'cartopy>=0.16.0',
    'numpy>=1.18.0',
    'pandas>=1.0.0',
    'beautifulsoup4>=4.9.1',
    'requests>=2.22.0'
    ]
)
