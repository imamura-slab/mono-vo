TARGET := calib
VER    := -std=c++11
LIB    := `pkg-config --cflags opencv` `pkg-config --libs opencv`
OPTION := -O2

default:
	g++ ${TARGET}.cpp ${VER} ${OPTION} ${LIB}

clean:
	rm -f *.o main

