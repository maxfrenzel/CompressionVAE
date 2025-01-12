import setuptools
import platform

with open("README.md", "r") as fh:
    long_description = fh.read()

# Determine the correct TensorFlow package based on architecture
if platform.machine() == 'arm64':  # Apple Silicon
    tensorflow_packages = [
        'tensorflow-macos==2.8.0',
        'tensorflow-metal==0.4.0'
    ]
else:  # Intel/AMD
    tensorflow_packages = ['tensorflow>=2.9.0,<2.10.0']

setuptools.setup(
    name="cvae",
    version="0.2.0",
    author="Max Frenzel",
    author_email="maxfrenzel+cvae@gmail.com",
    description="CompressionVAE: General purpose dimensionality reduction and manifold learning tool based on "
                "Variational Autoencoder.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxfrenzel/CompressionVAE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.5,<1.23.0',
        'matplotlib>=3.3.0,<4.0.0',
        'joblib>=1.0.0,<2.0.0',
        'tqdm>=4.50.0,<5.0.0',
        'pandas>=1.3.0,<2.0.0'
    ] + tensorflow_packages,
    extras_require={
        'test': ['scikit-learn>=1.0.0']
    },
    keywords='vae variational autoencoder manifold dimensionality reduction compression tensorflow'
)