#!/bin/bash

# Set the paths
EXPORT_PATH="/mnt/m2kingston/dev/godot-games/tails-of-ailurus/addons/godot-llama-cpp/lib/"

BUILD_TYPES=("Debug" "Release")

build_and_copy() {
    local BUILD_TYPE=$1
    local BUILD_PATH="godot_llama_cpp-$BUILD_TYPE"

    cmake -B$BUILD_PATH -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    cmake --build $BUILD_PATH --parallel -j16

    # Copy build directory to export path
    cp -r $BUILD_PATH/godot_llama_cpp/lib/* $EXPORT_PATH
}

cmake --version

# Run debug build
build_and_copy ${BUILD_TYPES[0]}

# Run release build
build_and_copy ${BUILD_TYPES[1]}

echo "Build and copy completed successfully."
