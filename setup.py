from setuptools import setup, find_packages

setup(
    name='PopGen',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pyyaml',

    ],
    entry_points={
        'console_scripts': [
            'popgen = PopGen.project:main',  # Assume project.py has main()
        ],
    },
    package_data={
        # 包含配置文件和示例数据集
        'PopGen': ['data/configuration_arizona.yaml', 'data/Arizona/*.csv'],
    },
    include_package_data=True,
)
