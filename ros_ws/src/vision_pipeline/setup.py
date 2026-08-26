import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'vision_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pieromutasci',
    maintainer_email='pieromutasci@gmail.com',
    description='Dataset-collection ROS2 nodes and YOLO train/eval/predict utilities for the TIAGo Pro vision pipeline.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_saver = vision_pipeline.camera_saver_node:main',
            'orbit_around_table = vision_pipeline.orbit_around_table_node:main',
            'teleop = vision_pipeline.teleop_node:main',
            'yolo_train = vision_pipeline.yolo.train:main',
            'yolo_evaluate = vision_pipeline.yolo.evaluate:main',
            'yolo_predict = vision_pipeline.yolo.predict:main',
            'yolo_models_comparison = vision_pipeline.yolo.models_comparison:main',
        ],
    },
)
