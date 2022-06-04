C = hipcc 
CFLAGS = -std=c++11

all: CNN

# mat: matrix.o main.o 
# 	$(C) $(CFLAGS) -o mat matrix.o main.o

# matrix.o: matrix.cpp 
# 	$(C) $(CFLAGS) -c matrix.cpp -o matrix.o
CNN: matmul.o CNN.o mat.o conv.o
	$(C) $(CFLAGS) -o CNN matmul.o CNN.o mat.o conv.o

CNN.o: CNN.cpp 
	$(C) $(CFLAGS) -c CNN.cpp -o CNN.o


conv: conv.o mat.o
	$(C) $(CFLAGS) -o conv conv.o mat.o
	
conv.o: conv2D.cpp  
	$(C) $(CFLAGS) -dc conv2D.cpp -o conv.o
	
maxpool: maxpool.o mat.o 
	$(C) $(CFLAGS) -o maxpool maxpool.o mat.o

maxpool.o: maxPool.cpp 
	$(C) $(CFLAGS) -dc maxPool.cpp -o maxpool.o

dense: dense.o mat.o 
	$(C) $(CFLAGS) -o dense dense.o mat.o

dense.o: dense.cpp 
	$(C) $(CFLAGS) -dc dense.cpp -o dense.o	

matmul.o: matmul.cpp
	$(C) $(CFLAGS) -dc matmul.cpp
mat.o: mat.cpp
	$(C) $(CFLAGS) -dc mat.cpp

clean:
	rm *.o conv, mat, main, CNN
