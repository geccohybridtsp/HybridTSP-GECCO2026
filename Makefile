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

# Run with default test
test: $(TARGET)
	./$(TARGET) ../data/berlin52.tsp --pool-size 20 --patience 5

# Run on larger instance
test-large: $(TARGET)
	./$(TARGET) ../data/pr264.tsp --pool-size 30 --patience 10

.PHONY: all clean debug test test-large
