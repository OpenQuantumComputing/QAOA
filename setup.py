from setuptools import setup, find_packages

setup(
    name="qaoa",
    version="0.2",
    license="GNU General Public License v3.0",
    author="Franz Georg Fuchs",
    author_email="franzgeorgfuchs@gmail.com",
    description="Quantum Alternating Operator Ansatz/Quantum Approximate Optimization Algorithm (QAOA)",
    long_description=open("README.md").read(),
    keywords="quantum computing, qaoa, qiskit",
    install_requires=[
        "numpy",
        "scipy",
        "structlog",
        "matplotlib",
        "networkx",
        "jupyter",
        "qiskit",
        "qiskit-aer",
        "qiskit-algorithms",
        "qiskit-finance",
        "pylatexenc",
    ],
)
