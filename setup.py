from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements=f.read().splitlines()
setup(
    name='medical_chatbot_rag',
    version='1.0.0',
    author='madhuri',
    packages=find_packages(),
    install_requires=requirements,
)    
    