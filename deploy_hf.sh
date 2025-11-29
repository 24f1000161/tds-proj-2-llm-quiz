#!/bin/bash
# Deploy to Hugging Face Spaces

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Login if needed
huggingface-cli whoami || huggingface-cli login

# Set your space name
SPACE_NAME="${1:-llm-quiz-solver}"
HF_USERNAME=$(huggingface-cli whoami | head -1)

echo "Deploying to: $HF_USERNAME/$SPACE_NAME"

# Create space if it doesn't exist
huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk docker 2>/dev/null || true

# Clone and push
TEMP_DIR=$(mktemp -d)
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" "$TEMP_DIR/space" 2>/dev/null || \
    git init "$TEMP_DIR/space"

# Copy files
cp -r quiz_solver "$TEMP_DIR/space/"
cp pyproject.toml "$TEMP_DIR/space/"
cp uv.lock "$TEMP_DIR/space/"
cp Dockerfile "$TEMP_DIR/space/"
cp .dockerignore "$TEMP_DIR/space/"
cp README_HF.md "$TEMP_DIR/space/README.md"

# Push
cd "$TEMP_DIR/space"
git add .
git commit -m "Deploy quiz solver"
git push "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" main

echo ""
echo "Deployed! Set secrets in HF Space settings:"
echo "  - STUDENT_SECRETS=email:secret"
echo "  - AIPIPE_TOKEN=your-token"
echo ""
echo "Space URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

# Cleanup
rm -rf "$TEMP_DIR"
