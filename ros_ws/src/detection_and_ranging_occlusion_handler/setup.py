from setuptools import find_packages, setup

package_name = 'detection_and_ranging_occlusion_handler'

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
    description='Real-time YOLO object detection fused with depth to estimate 3D object position, con gestione robusta delle occlusioni tra oggetti tracciati.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rt_object_detection_occlusion_handler = detection_and_ranging_occlusion_handler.rt_object_detection_node_occlusion_handler:main',
        ],
    },
)
