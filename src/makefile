ifeq ($(DEBUG), true)
	PARAM = -ggdb3 -Wall -std=c++11 -fopenmp
else
	PARAM = -O3 -DNDEBUG -std=c++11 -fopenmp
endif

LINKPARAM = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

main : main.cpp
	g++ main.cpp -o $@ $(PARAM) $(LINKPARAM)

clean :
	-rm *.o *.gch main

.PHONY : clean
