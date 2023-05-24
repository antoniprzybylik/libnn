.PHONY: all clean build debug
.SUFFIXES: .o .cpp

PROJECT = demo

CXX = g++
LINK = g++
STRIP = strip

CXXFLAGS = -MMD -MP -Wall -I../../libnn/ -std=c++20 -pedantic
LDFLAGS = -fPIC -L../../libnn/ -lnn

OBJS = \
main.o \
training.o \
simple_net.o \

DEPS = $(OBJS:%.o=%.d)

all: build

compile: $(OBJS)

build: CXXFLAGS += -O3
build: compile
	$(LINK) $(OBJS) -o $(PROJECT) $(LDFLAGS)
	$(STRIP) $(PROJECT)

debug: CXXFLAGS += -O0 -g
debug: clean | compile
	$(LINK) $(OBJS) -o $(PROJECT) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(DEPS)

.cpp.o:
	$(CXX) $< -o $@ -c $(CXXFLAGS)

-include $(DEPS)
