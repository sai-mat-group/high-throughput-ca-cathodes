#!/bin/bash

# cleanup_vasp.sh
# Traverses numbered subdirectories (00, 01, ...), deletes all files except
# POSCAR, CONTCAR, OSZICAR, and OUTCAR. Compresses OUTCAR and OSZICAR with bzip2.

KEEP_FILES=("POSCAR" "CONTCAR" "OSZICAR" "OUTCAR")

# Find subdirectories matching numeric names (00, 01, 02, ...)
for dir in */; do
    # Strip trailing slash
    dirname="${dir%/}"

    # Process only directories with numeric names
    if [[ ! "$dirname" =~ ^[0-9]+$ ]]; then
        continue
    fi

    echo "=========================================="
    echo "Processing directory: $dirname"
    echo "=========================================="

    # -------------------------------------------------------
    # Step 1: Delete all files that are NOT in the keep list
    # -------------------------------------------------------
    find "$dirname" -maxdepth 1 -type f | while read -r filepath; do
        filename=$(basename "$filepath")

        # Check if this file is in the keep list (also allow .bz2 variants)
        keep=false
        for kf in "${KEEP_FILES[@]}"; do
            if [[ "$filename" == "$kf" || "$filename" == "${kf}.bz2" ]]; then
                keep=true
                break
            fi
        done

        if [[ "$keep" == false ]]; then
            echo "  Deleting: $filepath"
            rm -f "$filepath"
        fi
    done

    # -------------------------------------------------------
    # Step 2: Compress OUTCAR and OSZICAR with bzip2 if not
    #         already compressed
    # -------------------------------------------------------
    for target in "OUTCAR" "OSZICAR"; do
        plain="$dirname/$target"
        zipped="$dirname/${target}.bz2"

        if [[ -f "$plain" ]]; then
            echo "  Compressing: $plain -> ${zipped}"
            bzip2 "$plain"          # produces <file>.bz2 and removes the original
        elif [[ -f "$zipped" ]]; then
            echo "  Already compressed: $zipped (skipping)"
        else
            echo "  Not found: $target or ${target}.bz2 (skipping)"
        fi
    done

    echo ""
done

echo "Done."
