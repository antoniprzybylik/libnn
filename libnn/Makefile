.PHONY: all clean build debug algebra
.SUFFIXES: .o .cpp

TARGET = libnn.a

CXX = g++

CXXFLAGS = -MMD -MP -std=c++20 -pedantic -Wall

OBJS = \
neuron.o \
constant.o \
sink.o \
sigmoid.o \
sum.o \
gelu.o \
softmax.o \
algebra/algebra.o \

LIBS = \
/usr/lib/libjsoncpp.a \

DEPS = $(OBJS:%.o=%.d)

all: build

algebra:
	$(MAKE) -C ./algebra/

compile: algebra $(OBJS)

build: CXXFLAGS += -O3
build: compile
	rm -f $(TARGET)
	ar -rcT $(TARGET) $(OBJS) $(LIBS)
	ranlib $(TARGET)

debug: CXXFLAGS += -O0 -DGLIBCXX_DEBUG -g
debug: clean | compile
	rm -f $(TARGET)
	ar -rcT $(TARGET) $(OBJS) $(LIBS)
	ranlib $(TARGET)

clean:
	rm -f $(OBJS) $(DEPS)

.cpp.o:
	$(CXX) $< -o $@ -c $(CXXFLAGS)

-include $(DEPS)
