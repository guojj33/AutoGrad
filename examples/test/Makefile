CXX := g++

HEADER_DIR := ../../include
BIN_DIR := ./bin

INCLUDE := -I $(HEADER_DIR)

all : bin test.out

bin :
	@mkdir -p $(BIN_DIR)

test.out : test.cpp
	$(CXX) $(INCLUDE) $^ -o $(BIN_DIR)/$@

clean :
	@rm -rf $(BIN_DIR)