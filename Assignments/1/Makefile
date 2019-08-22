all: p1 p2 p3

p1: program1.cpp
	g++ -o p1 program1.cpp `pkg-config --cflags --libs opencv`

p2: program2.cpp
	g++ -o p2 program2.cpp `pkg-config --cflags --libs opencv`

p3: program3.cpp
	g++ -o p3 program3.cpp `pkg-config --cflags --libs opencv`

clean:
	rm p1
	rm p2
	rm p3
