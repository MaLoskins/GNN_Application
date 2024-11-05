#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <root_directory> [<ignore_dir1> <ignore_dir2> ...]"
  exit 1
fi

# First argument is the root directory
root_directory="$1"
shift

# Remaining arguments are directories to ignore (if any)
ignore_dirs=("$@")

# Build the find command with ignore directories if provided
ignore_params=()
if [ "${#ignore_dirs[@]}" -gt 0 ]; then
  for ignore_dir in "${ignore_dirs[@]}"; do
    ignore_params+=(! -path "$root_directory/$ignore_dir/*")
  done
fi

# Find files and output to output.txt in the specified format
find "$root_directory" -type f "${ignore_params[@]}" | while read -r file; do
  echo "$file:" >> output.txt
  cat "$file" >> output.txt
  echo -e "\n\n" >> output.txt
done
