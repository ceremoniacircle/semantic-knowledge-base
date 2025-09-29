# Semantic Knowledge Base - Document Embedding CLI

A comprehensive Python CLI tool for the Semantic Knowledge Base project. This tool processes, embeds, and manages documents in Pinecone vector databases using OpenAI's text-embedding-3-small model, intelligently categorizing content across multiple file formats and organizing embeddings into semantic namespaces for efficient retrieval.

## 🎯 Overview

The `embed_docs_to_pinecone.py` script transforms various document types into searchable vector embeddings, automatically categorizing content into four specialized namespaces:

- **ops**: Operations (logistics, pricing, venue, booking)
- **legal**: Legal & compliance (privacy, licensing, terms)
- **medical**: Health & safety (screening, medications, contraindications)
- **program**: Program content (retreats, ceremonies, modalities, schedule)

## 🚀 Features

### Multi-Format Document Processing
- **Markdown (.md)**: Native markdown content parsing
- **JSON (.json)**: Website/CMS structured content with metadata extraction
- **PowerPoint (.ppt, .pptx)**: Slide text extraction with slide numbering
- **Word Documents (.doc, .docx)**: Paragraph-based content extraction
- **PDF (.pdf)**: Page-by-page text extraction
- **Text Files (.txt)**: Plain text processing

### Intelligent Content Categorization
- **Path-based Classification**: Automatic categorization using file paths and URL slugs
- **Keyword Scoring System**: Content analysis using category-specific keyword matching
- **Smart Fallback Logic**: Default categorization rules for edge cases

### Advanced Text Processing
- **Token-aware Chunking**: Uses tiktoken encoder for precise token counting
- **Overlapping Chunks**: Configurable overlap (default 80 tokens) for context preservation
- **Sentence Boundary Breaking**: Intelligent chunk splitting at sentence endings
- **Configurable Chunk Size**: Default 500 tokens per chunk, adjustable via CLI

### Enhanced Metadata Extraction
- **Category-specific Metadata**: Tailored metadata fields per content type
- **Source Tracking**: Complete file path and URL preservation
- **Hierarchical Navigation**: Heading path extraction for structured browsing
- **Timestamp Tracking**: Processing and update timestamps

### Interactive Configuration
- **Index Selection**: Dynamic Pinecone index discovery and selection
- **Namespace Configuration**: Flexible namespace selection (all, custom, or specific)
- **Default Settings**: Pre-configured defaults for common use cases
- **Dry-run Mode**: Preview processing without actual embedding

## 📋 Prerequisites

### Required Dependencies
```bash
pip install openai pinecone-client python-dotenv click python-pptx python-docx PyPDF2 tiktoken
```

### Environment Variables
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Pinecone Setup
- Create a Pinecone index (recommended name: `journeys-faq`)
- Dimension: 1536 (for text-embedding-3-small)
- Metric: cosine

## 🛠️ Installation & Setup

1. **Clone and navigate to the project directory**
2. **Install dependencies**:
   ```bash
   pip install openai pinecone-client python-dotenv click python-pptx python-docx PyPDF2 tiktoken
   ```
3. **Set up environment variables** in `.env` file
4. **Verify configuration**:
   ```bash
   python embed_docs_to_pinecone.py config
   ```

## 💻 Usage

The tool provides a comprehensive CLI interface with multiple commands:

### Basic Document Embedding
```bash
# Interactive mode with prompts
python embed_docs_to_pinecone.py embed docs/

# Use defaults (journeys-faq index, all namespaces)
python embed_docs_to_pinecone.py embed docs/ --auto-defaults

# Dry run to preview processing
python embed_docs_to_pinecone.py embed docs/ --dry-run

# Custom configuration
python embed_docs_to_pinecone.py embed docs/ --index-name my-index --max-tokens 300 --overlap-tokens 50
```

### Semantic Search
```bash
# Interactive namespace selection
python embed_docs_to_pinecone.py search "What day is the cold plunge?"

# Target specific namespace
python embed_docs_to_pinecone.py search "medication contraindications" --namespace medical

# Advanced search with filtering
python embed_docs_to_pinecone.py search "ceremony schedule" --namespace program --top-k 3 --score-threshold 0.8
```

### Index Statistics
```bash
# Show all namespace statistics
python embed_docs_to_pinecone.py stats

# Target specific index
python embed_docs_to_pinecone.py stats --index-name journeys-faq

# Show specific namespace stats
python embed_docs_to_pinecone.py stats --namespace ops
```

### Namespace Management
```bash
# Clear specific namespace (with confirmation)
python embed_docs_to_pinecone.py clear --namespace ops

# Skip confirmation prompt
python embed_docs_to_pinecone.py clear --namespace medical --yes

# Target specific index
python embed_docs_to_pinecone.py clear --namespace program --index-name my-index
```

### Configuration Check
```bash
# Verify environment and connections
python embed_docs_to_pinecone.py config
```

## 🏗️ Architecture

### Core Classes and Components

#### DocumentEmbedder Class
Main orchestrator class handling:
- **Initialization**: OpenAI and Pinecone client setup
- **Configuration Management**: Index and namespace selection
- **File Processing**: Multi-format document reading
- **Embedding Generation**: OpenAI API integration
- **Vector Operations**: Pinecone upsert and query operations

#### Key Methods

**File Processing Pipeline**:
- `read_file_content()`: Multi-format file reading with metadata
- `extract_sections_from_content()`: Content sectioning and categorization
- `chunk_text_by_tokens()`: Token-aware text chunking
- `categorize_content()`: Intelligent content classification

**Embedding Operations**:
- `generate_embedding()`: OpenAI embedding generation
- `process_documents()`: End-to-end document processing
- `upsert_to_pinecone()`: Batch vector uploads

**Search and Management**:
- `search_test()`: Semantic search functionality
- `verify_index_stats()`: Index statistics and monitoring
- `clear_namespace()`: Namespace cleanup operations

### Content Categorization System

The tool uses a sophisticated scoring system for content categorization:

```python
category_keywords = {
    "ops": {
        "paths": ["location", "venue", "contact", "booking", "pricing"],
        "keywords": ["address", "check-in", "deposit", "refund", "venue"]
    },
    "legal": {
        "paths": ["privacy-policy", "terms", "consent", "licensing"],
        "keywords": ["privacy", "consent", "licensing", "compliance"]
    },
    "medical": {
        "paths": ["health", "screening", "medical"],
        "keywords": ["screening", "medication", "contraindication"]
    },
    "program": {
        "paths": ["journeys", "program", "ceremonies"],
        "keywords": ["ceremony", "retreat", "meditation", "integration"]
    }
}
```

### Data Flow Architecture

1. **File Discovery**: Recursive scanning for supported file types
2. **Content Extraction**: Format-specific content parsing
3. **Section Splitting**: Logical content segmentation
4. **Categorization**: Multi-factor content classification
5. **Chunking**: Token-aware text segmentation
6. **Embedding**: OpenAI API vector generation
7. **Metadata Enhancement**: Category-specific metadata addition
8. **Storage**: Pinecone namespace-based organization

## 🔧 Configuration Options

### CLI Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--index-name` | Target Pinecone index | Interactive prompt | `--index-name journeys-faq` |
| `--batch-size` | Vectors per upsert batch | 100 | `--batch-size 50` |
| `--max-tokens` | Maximum tokens per chunk | 500 | `--max-tokens 300` |
| `--overlap-tokens` | Token overlap between chunks | 80 | `--overlap-tokens 50` |
| `--auto-defaults` | Skip interactive prompts | False | `--auto-defaults` |
| `--dry-run` | Preview without processing | False | `--dry-run` |

### Namespace Configuration

1. **All Default Namespaces**: Use all four category namespaces
2. **Selective Namespaces**: Choose specific categories to process
3. **Custom Namespaces**: Define custom namespace names and descriptions

### Category-Specific Metadata

#### Operations (ops)
- `dates_mentioned`: Extracted date patterns
- `deposit_amount`: Identified pricing information
- `has_location_info`: Location/address presence flag

#### Medical (medical)
- `contraindications`: Identified medication/health contraindications
- `requires_screening`: Health screening requirement flag
- `disclaimer_type`: "medical"

#### Legal (legal)
- `jurisdiction`: Identified legal jurisdiction (e.g., "CO")
- `licensing_scope`: License type identification
- `data_policy`: Privacy/data policy presence flag

#### Program (program)
- `program_type`: retreat/course/training classification
- `ceremonies_count`: Number of ceremonies mentioned
- `modalities`: Identified therapeutic modalities

## 🚨 Error Handling & Troubleshooting

### Common Issues

**Missing Dependencies**:
```bash
pip install tiktoken python-pptx python-docx PyPDF2
```

**Environment Variables Not Set**:
- Verify `.env` file exists and contains valid API keys
- Use `python embed_docs_to_pinecone.py config` to check configuration

**Pinecone Connection Issues**:
- Verify index name exists in Pinecone console
- Check API key permissions and quota limits

**File Processing Errors**:
- Ensure files are not corrupted or password-protected
- Check file encoding (UTF-8 recommended)

### Performance Optimization

**Large Document Sets**:
- Use `--batch-size 50` for slower connections
- Implement rate limiting for API quota management
- Consider processing subsets for initial testing

**Memory Management**:
- Process documents in batches for large datasets
- Monitor token counts for chunk size optimization

## 📊 Output Examples

### Embedding Process Output
```
🚀 Starting embedding process...
   📁 Source: docs/
   📊 Index: journeys-faq
   🧩 Max tokens per chunk: 500

📊 Current index statistics:
  Total vectors: 1,247

Found 92 files: 45 MD, 47 JSON, 0 other

Processing: docs/pages/journeys-awaken.json
  Extracted 1 sections
    Generating embedding for chunk 1/3 of 'Awaken Journey Program' -> namespace: program

📁 Processing namespace 'program': 156 vectors
   📝 Program content (retreats, ceremonies, modalities, schedule)
  ✅ Upserted batch 1/2 (100 vectors)
  ✅ Upserted batch 2/2 (56 vectors)

✅ Successfully embedded 342 document chunks!
```

### Search Results Output
```
🔍 Searching in index 'journeys-faq'
   📝 Query: 'What day is the cold plunge?'
   📁 Namespace: ops

📋 Found 3 results:

🎯 Result 1
   📊 Score: 0.8547
   📝 Page: journeys-awaken
   🏷️  Category: ops
   📍 Path: Journeys › Awaken Journey Program › Daily Schedule
   💬 Content: Cold plunge sessions are held every morning at 7 AM, starting on Day 2 of the retreat...
```

## 🔄 Maintenance & Updates

### Regular Tasks
1. **Monitor Index Statistics**: Check vector counts and namespace distribution
2. **Update Categories**: Adjust keyword patterns based on content evolution
3. **Refresh Embeddings**: Re-process updated documents periodically
4. **Backup Configurations**: Save successful index and namespace configurations

### Version Compatibility
- OpenAI API: Compatible with latest text-embedding-3-small model
- Pinecone: Supports both Pinecone v2 and v3 clients
- Python: Requires Python 3.7+

---

*Generated with Claude Code for the Semantic Knowledge Base project*