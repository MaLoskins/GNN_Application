#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <root_directory> [<ignore_dir1> <ignore_dir2> ...]"
  exit 1
fi

# Normalize paths by converting backslashes to forward slashes
normalize_path() {
  echo "$1" | sed 's|\\|/|g'
}

# First argument is the root directory (normalized)
root_directory=$(normalize_path "$1")
shift

# Remaining arguments are directories to ignore (if any)
ignore_dirs=("$@")

# Build the find command with ignore directories if provided
ignore_params=()
if [ "${#ignore_dirs[@]}" -gt 0 ]; then
  for ignore_dir in "${ignore_dirs[@]}"; do
    # Normalize each ignore directory path
    normalized_ignore_dir=$(normalize_path "$ignore_dir")
    ignore_params+=(! -path "$root_directory/$normalized_ignore_dir/*")
  done
fi

# Find files with specific extensions and output to output.txt in the specified format
find "$root_directory" -type f \( -name "*.js" -o -name "*.py" -o -name "*.css" \) "${ignore_params[@]}" | while IFS= read -r file; do
  echo "$file:" >> output.txt
  cat "$file" >> output.txt
  echo -e "\n\n" >> output.txt
done
