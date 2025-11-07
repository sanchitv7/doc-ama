#!/bin/bash
set -e  # Exit on error

echo "🚀 Setting up doc-ama workspace..."

# Check for uv package manager
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# Check for Python 3.13+
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Use uv sync to install dependencies based on pyproject.toml and uv.lock
# This creates/updates .venv automatically and installs all dependencies
echo "📦 Installing dependencies with uv sync..."
if [ -f "pyproject.toml" ]; then
    echo "Using pyproject.toml for dependency management..."
    uv sync
elif [ -f "requirements.txt" ]; then
    echo "Fallback: Using requirements.txt..."
    uv venv
    uv pip install -r requirements.txt
else
    echo "❌ Error: No pyproject.toml or requirements.txt found"
    exit 1
fi

# Copy .env file from root if it exists
if [ -f "$CONDUCTOR_ROOT_PATH/.env" ]; then
    echo "📋 Copying .env file from repository root..."
    cp "$CONDUCTOR_ROOT_PATH/.env" .env
else
    echo "⚠️  Warning: No .env file found in repository root"
    echo "   Please create one from .env.template and add your OPENROUTER_API_KEY"
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "   Created .env from template - you need to add your API key!"
    fi
fi

# Check for required environment variable
if [ -f ".env" ]; then
    if ! grep -q "OPENROUTER_API_KEY=sk-" .env 2>/dev/null; then
        echo "⚠️  Warning: OPENROUTER_API_KEY not set in .env file"
        echo "   The application will not work without a valid API key"
    else
        echo "✅ OPENROUTER_API_KEY found in .env"
    fi
fi

# Create ChromaDB directory if it doesn't exist
mkdir -p chroma_db

echo "✅ Workspace setup complete!"
echo ""
echo "Next steps:"
echo "  1. Ensure .env has your OPENROUTER_API_KEY"
echo "  2. Click 'Run' to start the application"
echo "  3. Open http://localhost:7860 in your browser"
