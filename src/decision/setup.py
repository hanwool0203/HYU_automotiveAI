from setuptools import find_packages, setup

package_name = 'decision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ircv7',
    maintainer_email='ircv7@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "sign_decision_node = decision.sign_decision:main",
            "roundabout_decision_node = decision.roundabout_decision:main",
            "combine_decision_node = decision.combine_decision:main",
            "acc_ttc_node = decision.accttc:main",
        ],
    },
)
