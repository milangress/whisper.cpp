#!/bin/bash

set -e

print_header() {
    echo "
╔════════════════════════════════════════════╗
║           Whisper CoreML Builder           ║
║        Metal Models for whisper.cpp        ║
╚════════════════════════════════════════════╝
"
}

print_section() {
    echo "
▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
           $1
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
"
}

# Usage: ./generate-and-upload-coreml-model.sh <model-name>
if [ $# -eq 0 ]; then
    print_header
    echo "No model name supplied"
    echo "Usage: ./generate-and-upload-coreml-model.sh <model-name>"
    echo "Available models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3"
    exit 1
fi

# Get Storj access grant from 1Password
STORJ_ACCESS=$(op read "op://Personal/model.milan.place-storij/storij-access-grant")
if [ -z "$STORJ_ACCESS" ]; then
    echo "Failed to get Storj access grant from 1Password"
    exit 1
fi

# Set up variables
STORJ_BUCKET="sj://models.milan.place/whisper-cpp/metal"
MODEL_NAME="$1"
WD=$(dirname "$0")
cd "$WD/../" || exit

print_header

# Function to generate, compile, and upload model
generate_and_upload_model() {
    local model_name="$1"
    local is_quantized="$2"
    local target_dir="${STORJ_BUCKET}${is_quantized:+/quantized}"
    
    print_section "${is_quantized:+Quantized }Model Generation"
    
    # Generate CoreML model
    python3 models/convert-whisper-to-coreml.py \
        --model "$model_name" \
        --encoder-only True \
        --optimize-ane True \
        ${is_quantized:+--quantize True} 2>&1 | tee "models/${model_name}.log"

    # Compile model
    xcrun coremlc compile "models/coreml-encoder-${model_name}.mlpackage" models/
    rm -rf "models/ggml-${model_name}-encoder.mlmodelc"
    mv -v "models/coreml-encoder-${model_name}.mlmodelc" "models/ggml-${model_name}-encoder.mlmodelc"

    # Zip the model
    (cd models && zip -r "ggml-${model_name}-encoder.mlmodelc.zip" "ggml-${model_name}-encoder.mlmodelc")

    # Upload to Storj
    print_section "Uploading ${is_quantized:+Quantized }Model"
    
    # Upload model zip
    uplink cp --access "$STORJ_ACCESS" --progress \
        "models/ggml-${model_name}-encoder.mlmodelc.zip" \
        "${target_dir}/ggml-${model_name}-encoder.mlmodelc.zip"

    # Create and upload download link file
    echo "https://models.milan.place/whisper-cpp/metal${is_quantized:+/quantized}/ggml-${model_name}-encoder.mlmodelc.zip" > "models/${model_name}.txt"
    uplink cp --access "$STORJ_ACCESS" \
        "models/${model_name}.txt" \
        "${target_dir}/${model_name}.txt"
    rm -f "models/${model_name}.txt"

    # Upload log file
    uplink cp --access "$STORJ_ACCESS" \
        "models/${model_name}.log" \
        "${target_dir}/${model_name}.log"
    rm -f "models/${model_name}.log"

    # Cleanup
    rm -f "models/ggml-${model_name}-encoder.mlmodelc.zip"
    rm -rf "models/ggml-${model_name}-encoder.mlmodelc"
    rm -rf "models/coreml-encoder-${model_name}.mlpackage"
}

echo "Generating and uploading regular model..."
generate_and_upload_model "$MODEL_NAME" ""

echo "Generating and uploading quantized model..."
generate_and_upload_model "$MODEL_NAME" "true"

print_section "Bucket Contents"
echo "Regular models:"
uplink ls --access "$STORJ_ACCESS" --recursive "${STORJ_BUCKET}"
echo -e "\nQuantized models:"
uplink ls --access "$STORJ_ACCESS" --recursive "${STORJ_BUCKET}/quantized"

print_section "Complete!"
echo "Models generated, uploaded, and cleaned up successfully!" 