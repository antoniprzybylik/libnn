.PHONY: all clean build debug
.SUFFIXES: .o .cpp

TARGET = libnn.a

CXX = g++

CXXFLAGS = -MMD -MP -std=c++20 -pedantic -Wall
LDFLAGS = -fPIC -lm

OBJS = \
neuron.o \
constant.o \
sink.o \
sigmoid.o \
sum.o \

DEPS = $(OBJS:%.o=%.d)

all: build

compile: $(OBJS)

build: CXXFLAGS += -O3
build: compile
	ar r $(TARGET) $(OBJS)
	ranlib $(TARGET)

debug: CXXFLAGS += -DGLIBCXX_DEBUG -g
debug: clean | compile
	ar r $(TARGET) $(OBJS)
	ranlib $(TARGET)

clean:
	rm -f $(OBJS) $(DEPS)

.cpp.o:
	$(CXX) $< -o $@ -c $(CXXFLAGS)

-include $(DEPS)
