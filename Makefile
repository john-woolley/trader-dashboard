# Define the source files
SRC_FILES := $(wildcard src/*.py)
MAIN_FILE := main.py

# Define the targets
.PHONY: format

# Target to format all the source files and main.py
format:
	ruff $(SRC_FILES) $(MAIN_FILE)
