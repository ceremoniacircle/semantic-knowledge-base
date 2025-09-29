# 🧠 Semantic Knowledge Base

An AI-powered semantic search system for organizational knowledge using OpenAI embeddings and Pinecone vector database. Transform your documents into an intelligent, searchable knowledge base with multi-format support and advanced categorization.

## ✨ Key Features

- **🔍 Semantic Search**: Find information using natural language queries, not just keyword matching
- **📁 Multi-Format Support**: Process Markdown, JSON, PowerPoint, Word, PDF, and text files
- **🏷️ Smart Categorization**: Automatically organize content into logical namespaces (ops, legal, medical, program)
- **🧩 Intelligent Chunking**: Token-aware text splitting with configurable overlap for optimal search results
- **🔒 Security-First**: Comprehensive .gitignore patterns protect sensitive information
- **⚡ Interactive CLI**: User-friendly prompts for index and namespace selection
- **📊 Rich Metadata**: Enhanced metadata extraction tailored to each content category

## 🎯 Use Cases

- **Knowledge Management**: Transform organizational documents into searchable databases
- **Customer Support**: Enable semantic search across help documentation and FAQs
- **Research & Development**: Make research papers and technical documents instantly searchable
- **Content Discovery**: Help teams find relevant information across large document collections
- **Compliance & Legal**: Organize and search legal documents, policies, and procedures

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Check configuration
python embed_docs_to_pinecone.py config

# Embed documents
python embed_docs_to_pinecone.py embed docs/

# Search documents
python embed_docs_to_pinecone.py search "What day is the cold plunge?"
```

## 📦 Installation

### Option 1: Direct Usage
```bash
git clone https://github.com/ceremoniacircle/semantic-knowledge-base.git
cd semantic-knowledge-base
pip install -r requirements.txt
```

### Option 2: Install as CLI Tool
```bash
pip install -e .
# Now you can use 'embedder' command anywhere
embedder config
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Verify Setup
```bash
embedder config
```

## 🛠️ CLI Commands

### 📄 Embed Documents
Embed markdown documents into Pinecone:

```bash
# Basic usage
embedder embed docs/

# Advanced options
embedder embed docs/retreats/ --namespace retreats --chunk-size 800
embedder embed . --dry-run  # Preview what would be processed
```

**Options:**
- `--index-name`: Pinecone index name (default: `faq`)
- `--namespace`: Pinecone namespace (default: `awaken-docs`)
- `--batch-size`: Batch size for upserts (default: `100`)
- `--chunk-size`: Maximum chunk size in characters (default: `1000`)
- `--overlap`: Overlap between chunks (default: `100`)
- `--dry-run`: Preview without actually embedding

### 🔍 Search Documents
Search for documents using semantic similarity:

```bash
# Basic search
embedder search "What day is the cold plunge?"

# Advanced search
embedder search "team structure" --top-k 3 --score-threshold 0.8
```

**Options:**
- `--index-name`: Pinecone index name (default: `faq`)
- `--namespace`: Pinecone namespace (default: `awaken-docs`)
- `--top-k`: Number of results to return (default: `5`)
- `--score-threshold`: Minimum similarity score (default: `0.0`)

### 📊 View Statistics
Show index statistics:

```bash
# All namespaces
embedder stats

# Specific namespace
embedder stats --namespace awaken-docs

# Different index
embedder stats --index-name my-index
```

### 🗑️ Clear Data
Clear vectors from a namespace:

```bash
# Interactive confirmation
embedder clear

# Skip confirmation
embedder clear --yes

# Specific namespace
embedder clear --namespace old-docs
```

### ⚙️ Check Configuration
Verify environment and connections:

```bash
embedder config
```

## 🎯 Usage Examples

### Claude AI Integration
Claude can use this tool directly:

```bash
# Embed new documents
embedder embed docs/new-content/ --namespace new-docs

# Search for specific information
embedder search "ceremony guidelines" --top-k 3

# Check what's in the index
embedder stats

# Clear old data before re-embedding
embedder clear --namespace old-docs --yes
```

### Batch Processing
```bash
# Process multiple directories
embedder embed docs/retreats/ --namespace retreats
embedder embed docs/ceremonies/ --namespace ceremonies
embedder embed docs/guides/ --namespace guides

# Search across all namespaces by checking each
embedder search "meditation practices" --namespace retreats
embedder search "meditation practices" --namespace ceremonies
```

### Development Workflow
```bash
# 1. Preview what would be processed
embedder embed docs/ --dry-run

# 2. Clear existing data
embedder clear --yes

# 3. Embed new content
embedder embed docs/

# 4. Test search functionality
embedder search "test query"

# 5. Check final statistics
embedder stats
```

## 🏗️ How It Works

### Document Processing
1. **Discovery**: Recursively finds all `.md` files in specified directory
2. **Sectioning**: Intelligently splits documents into logical sections based on headers and structure
3. **Chunking**: Breaks large sections into overlapping chunks for better coverage
4. **Categorization**: Auto-categorizes content (schedule, staff, ceremony, etc.)

### Embedding Generation
1. **Text Preparation**: Cleans and prepares text for embedding
2. **OpenAI API**: Uses `text-embedding-3-small` model (1536 dimensions)
3. **Error Handling**: Robust retry logic for API calls
4. **Batch Processing**: Efficient processing of multiple documents

### Pinecone Storage
1. **Vector Creation**: Converts embeddings to Pinecone format
2. **Metadata**: Rich metadata including title, category, source file, etc.
3. **Namespacing**: Organizes vectors by namespace for easy management
4. **Batch Upserts**: Efficient batch uploads for performance

### Search Functionality
1. **Query Embedding**: Converts search query to embedding
2. **Similarity Search**: Uses cosine similarity for matching
3. **Filtering**: Optional score thresholds and result limits
4. **Rich Results**: Returns content with metadata and similarity scores

## 📊 Output Examples

### Embedding Process
```
🚀 Starting embedding process...
   📁 Source: docs/
   🗂️  Index: faq
   📍 Namespace: awaken-docs

Processing: docs/retreats/Awaken Staff Schedule.md
  Extracted 8 sections
    Generating embedding for chunk 1/2 of 'Day 0: Arrival Process'...
    Generating embedding for chunk 2/2 of 'Day 0: Arrival Process'...

📊 Generated 15 vectors

Upserting 15 vectors to Pinecone index 'faq'
  Upserted batch 1/1 (15 vectors)

✅ Successfully embedded 15 document chunks!
```

### Search Results
```
🔍 Searching for: 'What day is the cold plunge?'

📋 Found 3 results:
─────────────────────────────────────────────────────────────────

🎯 Result 1
   📊 Score: 0.8542
   📝 Title: Cold Plunge Information
   🏷️  Category: activities
   📄 File: Awaken Staff Schedule
   💬 Content: The cold plunge activity occurs on Day 1 of the retreat at 2:15pm...
```

## 🔧 Technical Details

### Requirements
- Python 3.8+
- OpenAI API key
- Pinecone API key
- Existing Pinecone index (created separately)

### Dependencies
- `openai>=1.0.0`: OpenAI API client
- `pinecone-client>=3.0.0`: Pinecone vector database client
- `python-dotenv>=1.0.0`: Environment variable management
- `click>=8.0.0`: CLI framework

### File Structure
```
semantic-knowledge-base/
├── embed_docs_to_pinecone.py  # Main CLI tool
├── requirements.txt           # Dependencies
├── setup.py                  # Installation script
├── .env.example             # Environment template
├── README.md               # This file
├── scripts/                # Documentation and utilities
└── docs/                   # Your documents (add your own content here)
    ├── .gitkeep           # Preserves folder structure
    └── journeys/          # Example structure for organization
```

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**
```bash
❌ OPENAI_API_KEY environment variable not set
💡 Set it in .env file or export OPENAI_API_KEY=your_key
```
→ Check your `.env` file and API keys

**Connection Issues**
```bash
❌ Connection error: 401 Unauthorized
```
→ Verify your API keys are correct and active

**No Results Found**
```bash
❌ No results found above score threshold 0.8
```
→ Try lowering the `--score-threshold` or check if documents are embedded

**Index Not Found**
```bash
❌ Index 'faq' not found
```
→ Create the Pinecone index first or specify correct `--index-name`

### Debug Mode
Add verbose output by modifying the script or using:
```bash
# Check what files would be processed
embedder embed docs/ --dry-run

# Verify index contents
embedder stats

# Test with simple query
embedder search "test" --top-k 1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check this README
2. Run `embedder config` to verify setup
3. Try `embedder --help` for command help
4. Open an issue on GitHub

---

**Made with ❤️ for semantic document search**