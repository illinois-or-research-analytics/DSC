# Compiler settings
#CXX = g++
#CC = gcc

CXXFLAGS = -std=c++17 -O2 -Iexternal/exactDSP-cpp -fno-lto
#CFLAGS = -O4 -Wall -march=native -flto -Iexternal/exactDSP-cpp -DNO_MAIN -DNDEBUG -DCUT_ONLY
CFLAGS = -O4 -Wall -march=native -Iexternal/exactDSP-cpp -DNO_MAIN -DNDEBUG -DCUT_ONLY -fno-lto

# Forced include for parser.c to get types like node, arc
FORCE_INCLUDE = \
  -include stdio.h \
  -include stdlib.h \
  -include string.h \
  -include stddef.h \
  -include external/exactDSP-cpp/hi_pr.h

# Source files
CPP_SRCS = main.cpp graph.cpp pushrelabel.cpp
C_SRCS = external/exactDSP-cpp/hi_pr.c external/exactDSP-cpp/timer.c

# Object files
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
C_OBJS = $(C_SRCS:.c=.o)

# Executable name
TARGET = flow

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(CPP_OBJS) $(C_OBJS)
	$(CXX) $(CPP_OBJS) $(C_OBJS) -o $(TARGET)

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C source files with forced include
external/exactDSP-cpp/%.o: external/exactDSP-cpp/%.c
	$(CC) $(CFLAGS) $(FORCE_INCLUDE) -c $< -o $@

# Clean target
clean:
	rm -f $(CPP_OBJS) $(C_OBJS) $(TARGET)

.PHONY: all clean
