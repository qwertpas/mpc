CC = g++
CFLAGS = \
	-I../include \
	-I../include/eigen \
	-I../include/qpOASES \

all: main

main: main.cpp
	$(CC) $(CFLAGS) -o main main.cpp

clean:
	rm -f main