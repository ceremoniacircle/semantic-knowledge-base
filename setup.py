#!/usr/bin/env python3
"""
Setup script for Document Embedder CLI Tool
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CLI tool to embed markdown documents using OpenAI and Pinecone"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'openai>=1.0.0',
        'pinecone-client>=3.0.0',
        'python-dotenv>=1.0.0',
        'click>=8.0.0'
    ]

setup(
    name="doc-embedder",
    version="1.0.0",
    author="Ceremonia AI",
    author_email="hello@ceremonia.ai",
    description="CLI tool to embed markdown documents using OpenAI text-embedding-3-small and Pinecone",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/doc-embedder",
    py_modules=["embed_docs_to_pinecone"],
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'embedder=embed_docs_to_pinecone:cli',
            'doc-embedder=embed_docs_to_pinecone:cli',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Utilities",
    ],
    keywords="embeddings, openai, pinecone, cli, markdown, documents, semantic-search",
    project_urls={
        "Bug Reports": "https://github.com/your-org/doc-embedder/issues",
        "Source": "https://github.com/your-org/doc-embedder",
        "Documentation": "https://github.com/your-org/doc-embedder#readme",
    },
)