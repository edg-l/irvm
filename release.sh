#!/usr/bin/env bash

cargo publish || exit

cd irvm-lower/ || exit
cargo publish || exit
cd ..
