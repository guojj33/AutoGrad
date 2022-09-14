CXX := g++

HEADER_DIR := ../../include
BIN_DIR := ./bin
OUTPUT_DIR := ./output

INCLUDE := -I $(HEADER_DIR)

all : bin output ExampleBoston.out

bin :
	@mkdir -p $(BIN_DIR)

output :
	@mkdir -p $(OUTPUT_DIR)

ExampleBoston.out : ExampleBoston.cpp
	$(CXX) $(INCLUDE) $^ -o $(BIN_DIR)/$@

clean :
	@rm -rf $(BIN_DIR)
	@rm -rf $(OUTPUT_DIR)