CXX := g++

HEADER_DIR := ../../include
BIN_DIR := ./bin
OUTPUT_DIR := ./output

INCLUDE := -I $(HEADER_DIR)

all : bin output ExampleXOR.out

bin :
	@mkdir -p $(BIN_DIR)

output :
	@mkdir -p $(OUTPUT_DIR)

ExampleXOR.out : ExampleXOR.cpp
	$(CXX) $(INCLUDE) $^ -o $(BIN_DIR)/$@

clean :
	@rm -rf $(BIN_DIR)
	@rm -rf $(OUTPUT_DIR)