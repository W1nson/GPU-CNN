C = g++
CFLAGS = -std=c++11

all: mat

mat: matrix.o main.o 
	$(C) $(CFLAGS) -o mat matrix.o main.o

matrix.o: matrix.cpp 
	$(C) $(CFLAGS) -c matrix.cpp -o matrix.o

main.o: main.cpp 
	$(C) $(CFLAGS) -c main.cpp -o main.o

clean:
	rm *.o mat 
