.PHONY: all clean build debug
.SUFFIXES: .o .cpp

TARGET = algebra.o

CXX = g++
LINK = ld

CXXFLAGS = -MMD -MP -Wall -std=c++20 -pedantic

OBJS = \
matrix.o \
vector.o \

DEPS = $(OBJS:%.o=%.d)

all: build

compile: $(OBJS)

build: CXXFLAGS += -O3
build: compile
	$(LINK) $(OBJS) -o $(TARGET) -relocatable

debug: CXXFLAGS += -g
debug: clean | compile
	$(LINK) $(OBJS) -o $(TARGET) -relocatable

clean:
	rm -f $(OBJS) $(DEPS)

.cpp.o:
	$(CXX) $< -o $@ -c $(CXXFLAGS)

-include $(DEPS)
