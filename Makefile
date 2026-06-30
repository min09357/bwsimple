CXX      = g++
CXXFLAGS = -O3 -march=native -mavx512f -std=c++20 -Wall -Wextra -pthread

TARGETS = randread_bw stream_bw

.PHONY: all clean

all: $(TARGETS)

randread_bw: randread_bw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

stream_bw: stream_bw.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
