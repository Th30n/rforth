#!/bin/bash

cat ./src/rforth.fs - | cargo run --release -- $@
