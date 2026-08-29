import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'pose_optimizer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pieromutasci',
    maintainer_email='pieromutasci@gmail.com',
    description='IK pose optimization nodes (MoveIt + KDL) for grasping targets detected by detection_and_ranging. Ogni nodo prende il nome dal criterio di scelta tra i candidati che ottimizza.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_margin_optimizer = pose_optimizer.joint_margin_optimizer:main',
        ],
    },
)
