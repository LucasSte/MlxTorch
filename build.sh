#!/bin/zsh

set -e

MLX_REPO="$PWD"/mlx_srcs

if [ ! -d "$MLX_REPO" ]; then
  git clone https://github.com/LucasSte/mlx.git --branch release-version mlx_srcs
fi
pushd mlx_srcs

BUILD_FOLDER="$PWD"/build

if [ ! -d "$BUILD_FOLDER" ]; then
  mkdir -p build
fi

cd build

MLX_LIB="$PWD"/libmlx.a

if [ ! -f "$MLX_LIB" ]; then
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j
fi
cd ../..
cp -r mlx_srcs/mlx ./

python3 setup.py "$1"