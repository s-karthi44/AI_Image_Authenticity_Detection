from setuptools import setup, find_packages

setup(
    name="ai_image_authenticity_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "scipy",
        "torch",
        "torchvision",
        "fastapi",
        "streamlit",
    ],
)
