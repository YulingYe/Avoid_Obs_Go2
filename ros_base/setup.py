from setuptools import find_packages, setup

setup(
    name="ros_base",
    version="0.0.1",
    author="Shiyi Chen",
    author_email="",
    license="",
    packages=find_packages(),
    description="Basic ROS node management module",
    python_requires=">=3.6",
    install_requires=[
        # "rclpy",
    ],
)
