CXX := g++

HEADER_DIR := ./include
BIN_DIR := ./bin
SRC_DIR := ./src

INCLUDE := -I $(HEADER_DIR)

all : bin autograd.out

bin :
	@mkdir -p $(BIN_DIR)

autograd.out : $(SRC_DIR)/main.cpp
	$(CXX) $(INCLUDE) $^ -o $(BIN_DIR)/$@

clean :
	@rm -rf $(BIN_DIR)