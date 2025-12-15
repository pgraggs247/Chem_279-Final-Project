#!/bin/bash

# path to the executable
EXECUTABLE="./build/normal_modes"

# path to the input directory json files
INPUT_DIR="sample_input/molecules/"

# check if the executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found: $EXECUTABLE"
    echo "Please build the executable first"
    echo "Usage: ./build.sh"
    exit 1
fi

# if a molecule name is provided, run only that molecule
if [ $# -eq 1 ]; then
    MOLECULE_NAME=$1
    CONFIG_FILE="$INPUT_DIR/$MOLECULE_NAME.json"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file not found: $CONFIG_FILE"
        echo "Available molecules:"
        ls -1 "$INPUT_DIR" | sed 's/.json//'
        exit 1
    fi

    echo "Running $MOLECULE_NAME ..."
    $EXECUTABLE "$MOLECULE_NAME"
    echo "Done"
    exit 0

# if no molecule name is provided, run all molecules in the input directory
elif [ $# -eq 0 ]; then
    echo "Running all molecules in the input directory: $INPUT_DIR ..."
    echo "========================================================"
        
    for config in "$INPUT_DIR"/*.json; do
        MOLECULE_NAME=$(basename "$config" .json)

        echo ""
        echo ">>> Processing: $MOLECULE_NAME ..."
        echo "--------------------------------------------------------"
        $EXECUTABLE "$MOLECULE_NAME"
    done
        
    echo ""
    echo "========================================================"
    echo "All molecules processed successfully"
    exit 0

else
    echo "Usage: $0 [molecule_name]"
    echo ""
    echo "Examples:"
    echo "  $0 H2O    # Run only water"
    echo "  $0        # Run all molecules"
    echo ""
    echo "Available molecules:"
    ls -1 "$INPUT_DIR" | sed 's/.json$//'
    exit 1
fi