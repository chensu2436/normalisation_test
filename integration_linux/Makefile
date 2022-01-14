
CC := gcc
NAME := normalization

INCLUDE=-I/usr/include/opencv4 
CFLAGS=-Wall -g
LIBS=-L.. -L/usr/lib/x86_64-linux-gnu -lstdc++ -lm -lpthread -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
#OBJ = main.o 

all: test

test: $(NAME)_test
  
$(NAME)_test: 
	$(CC) -std=c++11 -Wl,-rpath=.,-rpath=.. main.cpp -o $@ $(CFLAGS) $(INCLUDE) $(LIBS)

clean:
	$(RM) $(NAME)_test *.o
