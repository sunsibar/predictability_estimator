from setuptools import setup, find_packages

# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith('#')]
    
setup(
    name='predictability_estimator',
    version='0.1',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    # [ 'torch',  # Add other dependencies here
    # ],

    url='https://github.com/sunsibar/predictability_estimator',
    # author='Your Name',
    # author_email='your.email@example.com',
    description='A tiny package to fit MLP models from 1D features to 1D features.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)

