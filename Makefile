# makefile for building and running the application

.PHONY: all cutils ctrain cpostman converter run clean

# default target to build all components
all: cutils ctrain cpostman

# build Cython extensions in restful/cutils
cutils:
	@echo "Building Cython extensions in restful/cutils..."
	@cd restful/cutils && python setup.py build_ext --inplace


# build Cython extensions in training
ctrain:
	@echo "Building Cython extensions in training..."
	@cd training && python setup.py build_ext --inplace


# build Cython extensions in postman
cpostman:
	@echo "Building Cython extensions in postman..."
	@cd postman && python setup.py build_ext --inplace


# run the converter script
converter:
	@echo "Running converter script..."
	@python postman/converter.py


# run the application with uvicorn
run:
	@echo "Starting the application..."
	@uvicorn app:app --host 0.0.0.0 --port 7860 --reload


# clean up build artifacts
clean:
	@echo "Cleaning up build artifacts..."
	@find . -type f -name "*.so" -delete
	@find . -type f -name "*.c" -delete
	@find . -type f -name "*.cpp" -delete
	@find . -type d -name "__pycache__" -exec rm -r {} +


# help message
help:
	@echo "Usage:"
	@echo "  make all       - Build all Cython extensions"
	@echo "  make cutils    - Build Cython extensions in restful/cutils"
	@echo "  make ctrain    - Build Cython extensions in training"
	@echo "  make cpostman  - Build Cython extensions in postman"
	@echo "  make converter - Run the converter script"
	@echo "  make run       - Start the application with Uvicorn"
	@echo "  make clean     - Remove build artifacts"
