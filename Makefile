# Makefile for Hybrid TSP Solver

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
TARGET = hybrid_tsp
SOURCES = hybrid_tsp.cpp
HEADERS = hybrid_tsp.h

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Debug build
debug: CXXFLAGS += -g -DDEBUG -O0
debug: clean $(TARGET)


.PHONY: all clean debug test test-large
