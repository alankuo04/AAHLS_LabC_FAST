#define MAX_HEIGHT 128
#define MAX_WIDTH 128

#define FILTER_SIZE 7

#include <iostream>
#include <math.h>
#include "hls_stream.h"

using namespace std;

typedef int DTYPE;

struct Window {
	DTYPE pix[FILTER_SIZE][FILTER_SIZE];
};

void fast_accel(DTYPE* img_in, int threshold, DTYPE* img_out, int rows, int cols);
