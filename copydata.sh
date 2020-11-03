#!/bin/bash

# copies data into binary folder 
# Set this script as post-compile hook, to keep things updated

# Remove old data
rm bin/debug/data -r
rm bin/release/data -r

# Copy new data
cp data bin/debug -r 
cp data bin/release -r 