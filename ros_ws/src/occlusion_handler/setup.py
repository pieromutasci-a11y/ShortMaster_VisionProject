from setuptools import find_packages, setup

package_name = 'occlusion_handler'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pieromutasci',
    maintainer_email='pieromutasci@gmail.com',
    description='Occlusion-aware YOLO+depth object detection, with multi-point surface sampling for grasp-circle fitting.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_and_occlusion_handler = occlusion_handler.detection_and_occlusion_handler:main',
        ],
    },
)
