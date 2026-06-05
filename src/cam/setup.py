import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'cam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
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
            'dual_cam_pub_node = cam.dual_cam_pub:main',
            'dual_cam_pub_undistorted_node = cam.dual_cam_pub_undistorted:main',
            'bev_node = cam.bev:main',
            'lane_seg_node = cam.lane_multiclass_trt:main',
            'stereo_depth_node = cam.stereo_depth:main',
            'depth_check_node = cam.depth_check:main',
            'yolo_node = cam.yolo:main',
        ],
    },
)
