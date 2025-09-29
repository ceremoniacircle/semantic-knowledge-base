# Document Embedder CLI Tool - Makefile

.PHONY: help install dev test clean embed search stats config

# Default target
help:
	@echo "📚 Document Embedder CLI Tool"
	@echo ""
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  dev         Install in development mode"
	@echo "  test        Run basic tests"
	@echo "  clean       Clean up temporary files"
	@echo "  embed       Embed documents (docs/ folder)"
	@echo "  search      Search documents (interactive)"
	@echo "  stats       Show index statistics"
	@echo "  config      Check configuration"

# Install dependencies
install:
	pip install -r requirements.txt

# Install in development mode
dev:
	pip install -e .

# Run basic functionality tests
test:
	@echo "🧪 Testing configuration..."
	python embed_docs_to_pinecone.py config
	@echo ""
	@echo "🧪 Testing dry run..."
	python embed_docs_to_pinecone.py embed docs/ --dry-run

# Clean up
clean:
	rm -rf __pycache__/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Embed documents
embed:
	@echo "📄 Embedding documents from docs/ folder..."
	python embed_docs_to_pinecone.py embed docs/

# Interactive search
search:
	@echo "🔍 Enter your search query:"
	@read -p "Query: " query; \
	python embed_docs_to_pinecone.py search "$$query"

# Show statistics
stats:
	@echo "📊 Index Statistics:"
	python embed_docs_to_pinecone.py stats

# Check configuration
config:
	@echo "⚙️ Configuration Check:"
	python embed_docs_to_pinecone.py config

# Quick setup for new users
setup:
	@echo "🚀 Setting up Document Embedder..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from template..."; \
		cp .env.example .env; \
		echo "✅ Please edit .env with your API keys"; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "🧪 Testing configuration..."
	python embed_docs_to_pinecone.py config
	@echo ""
	@echo "🎉 Setup complete! Try: make embed"