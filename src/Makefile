TARGET = visodo
VER    = -std=c++11
LIB    = `pkg-config --cflags opencv` `pkg-config --libs opencv`

default: all

all: 
	g++ ${VER} ${OPTION} ${LIB} ${TARGET}.cpp

clean:
	rm -f *o

