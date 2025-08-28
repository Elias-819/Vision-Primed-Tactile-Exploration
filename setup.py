from setuptools import setup, find_packages

setup(
    name="AcTExplore",
    version="0.1.0",
    description="Active Tactile Exploration framework with integrated PCN",
    author="Your Name",
    author_email="your.email@example.com",
    # 找到自己写的包 + 把 pcn_pytorch 加进去
    packages=find_packages() + ["pcn_pytorch"],
    # 告诉 setuptools 源码实际在哪
    package_dir={
        "": ".",                      # 根目录也是包搜索起点
        "pcn_pytorch": "externals/pcn_pytorch"
    },
    include_package_data=True,
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.18.0",
        "open3d>=0.10.0",
        "tqdm>=4.0.0",
    ],
    python_requires=">=3.8",
)
