ifeq ($(DEBUG), true)
	PARAM = -ggdb3 -Wall -std=c++11 -fopenmp
else
	PARAM = -O3 -DNDEBUG -std=c++11 -fopenmp
endif

LINKPARAM = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

main : main.o ui.o
	g++ main.o ui.o -o main $(PARAM) $(LINKPARAM)

main.o : main.cpp ui.h
	g++ main.cpp -c $(PARAM)

ui.o : ui.cpp ui.h
	g++ ui.cpp -c $(PARAM)

clean :
	-rm *.o *.gch main

.PHONY : clean
