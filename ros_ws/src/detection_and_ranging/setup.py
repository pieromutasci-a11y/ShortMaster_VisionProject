import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'detection_and_ranging'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pieromutasci',
    maintainer_email='pieromutasci@gmail.com',
    description='Real-time YOLO object detection fused with depth to estimate 3D object position.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rt_object_detection = detection_and_ranging.rt_object_detection_node:main',
            'center_computation = detection_and_ranging.center_computation:main',
        ],
    },
)
