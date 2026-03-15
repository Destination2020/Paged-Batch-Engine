#!/bin/bash

# 遍历所有需要更新的文件类型
# C++ 和 CUDA 文件添加 // 注释
for file in $(find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) | grep -v build); do
    if [ -f "$file" ]; then
        # 检查第一行是否已经有这个注释
        if ! head -1 "$file" | grep -q "Updated on March 15, 2026"; then
            # 在文件开头添加注释
            sed -i '1s/^/\/\/ Updated on March 15, 2026\n/' "$file"
        fi
    fi
done

# Python 和 CMakeLists 文件添加 # 注释
for file in $(find . -type f \( -name "*.py" -o -name "CMakeLists.txt" -o -name "*.cmake" \) | grep -v build); do
    if [ -f "$file" ]; then
        # 检查第一行是否已经有这个注释
        if ! head -1 "$file" | grep -q "Updated on March 15, 2026"; then
            # 在文件开头添加注释
            sed -i '1s/^/# Updated on March 15, 2026\n/' "$file"
        fi
    fi
done

echo "All files updated successfully!"
