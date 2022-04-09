from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="GenerateCoherentText",  # Required
    version="1.0",  # Required
    description="Generate coherent text using semantic embedding, common sense templates and Monte-Carlo tree search methods",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/play0137/Generate_coherent_text",  # Optional
    author="Ying-Ren Chen",  # Optional
    author_email="play0137@gmail.com",  # Optional
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved ::  GPL-3.0 License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="Natural Language Generation, ConceptNet, commonsense knowledge, word embedding, Monte-Carlo Tree Search",  # Optional
    packages=find_packages(),  # Required
    python_requires=">=3.9, <4",
    install_requires=["keras",
                      "keras-gpu",
                      "keras-layer-normalization",
                      "tensorflow",
                      "tensorflow-gpu",
                      "gensim",
                      "numpy",
                      "jieba"],
)
