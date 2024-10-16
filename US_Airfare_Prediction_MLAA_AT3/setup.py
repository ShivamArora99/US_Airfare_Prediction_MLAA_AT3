from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='building a data product to help users in the USA estimate airfare for local travel.  Users input their trip details (origin, destination, date, time, cabin class) and the model predicts the fare price and displays all the information on a streamlit application.',
    author='Shivam Arora',
    license='',
)
