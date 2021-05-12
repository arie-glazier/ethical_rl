from setuptools import setup, find_packages

setup(
    name='ethical_rl',
    version='0.0.1',
    description='Ethical Reinforcement Learning Library',
    url='https://github.com/arie-glazier/ethical_rl',
    author='Arie Glazier',
    author_email='adglazier@gmail.com',
    license='unlicense',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
      'tensorflow>=2.3.0',
      'gym==0.17.2',
      'numpy==1.17.4',
      'gym-minigrid==1.0.1',
      'matplotlib==3.3.1',
      'gym[atari]'
    ]
)