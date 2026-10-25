from setuptools import find_packages, setup

setup(
    name="driver_monitoring_system",
    version="1.0.0",
    description="Real-Time Multi-Task Driver Monitoring System with Personalized Driver Safety Index",
    author="Akkiraju Pruthvi (22BCE2098, VIT Vellore)",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    include_package_data=True,
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.16.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.14.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "scikit-learn>=1.3.0"],
        "tflite": ["tensorflow>=2.14.0", "onnx2tf>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "dms-train=train:main",
            "dms-eval=evaluate:main",
            "dms-bench=benchmark:main",
            "dms-demo=inference_demo:main",
            "dms-dashboard=src.dashboard.app:main",
        ],
    },
)
