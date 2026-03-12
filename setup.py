from setuptools import setup, find_packages

setup(
    name="qaoa",
    version="2.0.0",
    license="GNU General Public License v3.0",
    author="Franz Georg Fuchs",
    author_email="franzgeorgfuchs@gmail.com",
    description="Quantum Alternating Operator Ansatz/Quantum Approximate Optimization Algorithm (QAOA)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples", "images"]),
    keywords="quantum computing, qaoa, qiskit",
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "structlog",
        "matplotlib",
        "networkx",
        "jupyter",
        "qiskit>=2.3.0",
        "qiskit-aer>=0.17.0",
        "qiskit-algorithms>=0.4.0",
        "qiskit-finance",
        "pylatexenc",
    ],
    extras_require={
        "dev": [
            "twine>=6.0.0",
            "build>=1.0.0",
            "pytest>=7.0.0",
        ],
    },
)
