#!/bin/bash

echo "🔍 Searching for 'nltk' import or usage in .py files..."

grep -nr --include="*.py" 'import nltk' . | grep -v "venv" | grep -v "__pycache__"

echo "✅ Done."

