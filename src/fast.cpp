#include "fast.h"

void ReadFromMem(int rows, int cols, DTYPE* src, hls::stream<DTYPE> &pixel_stream, hls::stream<DTYPE> &middle_stream)
{
	const int max_iterations = MAX_HEIGHT * MAX_WIDTH;
	read_image: for(int i=0; i<rows*cols; i++) {
#pragma HLS LOOP_TRIPCOUNT max=max_iterations
#pragma HLS PIPELINE II=1
		DTYPE pix = src[i];
		pixel_stream.write(pix);
		middle_stream.write(pix);

	}
}

void Window2D(int rows,  int cols, hls::stream<DTYPE> &pixel_stream, hls::stream<Window> &window_stream)
{
	DTYPE LineBuffer[FILTER_SIZE-1][MAX_WIDTH];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

	Window window;

	int col_ptr = 0;
	int ramp_up = cols * ((FILTER_SIZE-1)/2) + (FILTER_SIZE-1)/2;
	int num_pixels = rows * cols;
	int num_iterations = num_pixels + ramp_up;
	int width = rows-1;

	const int max_iterations = MAX_HEIGHT * MAX_WIDTH + MAX_HEIGHT * ((FILTER_SIZE-1)/2) + (FILTER_SIZE-1)/2;

	update_window: for(int n=0; n<num_iterations; n++) {
#pragma HLS LOOP_TRIPCOUNT max=max_iterations
#pragma HLS PIPELINE II=1
		DTYPE new_pixel = (n<num_pixels) ? pixel_stream.read() : 0;

		for(int i=0; i<FILTER_SIZE; i++) {
			for (int j=0; j<FILTER_SIZE-1; j++) {
				window.pix[i][j] = window.pix[i][j+1];
			}
			window.pix[i][FILTER_SIZE-1] = (i<FILTER_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
		}

		for(int i=0; i<FILTER_SIZE-2; i++) {
			LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
		}
		LineBuffer[FILTER_SIZE-2][col_ptr] = new_pixel;

		int temp = col_ptr + 1;
		int temp2 = col_ptr - width;
		if (temp2 == 0)
			col_ptr = 0;
		else
			col_ptr = temp;

		if (n >= ramp_up)
			window_stream.write(window);
	}
}

void FastCalc(int rows, int cols, hls::stream<DTYPE> &middle_stream, hls::stream<Window> &window_stream, int threshold, hls::stream<DTYPE> &output_stream)
{
	fast_filter: for(int y=0; y<rows; y++) {
#pragma HLS PIPELINE II=1
		for (int x=0; x<cols; x++) {
			Window window = window_stream.read();
			DTYPE middle_pixel = middle_stream.read();

			int out = 0;
			int four = 0;
			for (int i=0; i<FILTER_SIZE; i++) {
				for (int j=0; j<FILTER_SIZE; j++) {
					DTYPE pixel;
					int xoffset = (x+j-(FILTER_SIZE/2));
					int yoffset = (y+i-(FILTER_SIZE/2));

					if ((xoffset<0) || (xoffset>=cols) || (yoffset<0) || (yoffset>=rows)) {
						pixel = middle_pixel;
						four -= 10;
					}
					else
						pixel = window.pix[i][j];

					DTYPE temp = (abs(pixel - middle_pixel)) > threshold;
					if ((i==0 && j==3) || (i==0 && j==4) || (i==1 && j==5) || (i==2 && j==6) || (i==3 && j==6) || (i==4 && j==6) || (i==5 && j==5) || (i==6 && j==4) || (i==6 && j==3) || (i==6 && j==2) || (i==5 && j==1) || (i==4 && j==0) || (i==3 && j==0) || (i==2 && j==0) || (i==1 && j==1) || (i==0 && j==2)) {
						out += temp;
						if ((i==0 && j==3) || (i==3 && j==6) || (i==6 && j==3) || (i==3 && j==0))
							four += temp;
					}
				}
			}
			DTYPE output = (four >= 3 && out >= 12) ? 255 : 0;
			output_stream.write(output);
		}
	}
}

void WriteToMem(int rows, int cols, hls::stream<DTYPE> &output_stream, DTYPE *img_out)
{
	const int max_iterations = MAX_HEIGHT * MAX_WIDTH;
	write_image: for(int i=0; i<rows*cols; i++) {
#pragma HLS LOOP_TRIPCOUNT max=max_iterations
#pragma HLS PIPELINE II=1
		DTYPE pixel = output_stream.read();
		img_out[i] = pixel;
	}
}

void fast_accel(DTYPE* img_in, int threshold, DTYPE* img_out, int rows, int cols)
{

#pragma HLS INTERFACE m_axi port=img_in bundle=in depth=16384
#pragma HLS INTERFACE m_axi port=img_out bundle=out depth=16384
#pragma HLS INTERFACE s_axilite port=img_in bundle=control
#pragma HLS INTERFACE s_axilite port=img_out bundle=control
#pragma HLS INTERFACE s_axilite port=rows bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=threshold bundle=control

#pragma HLS DATAFLOW

	hls::stream<DTYPE> pixel_stream;
	hls::stream<DTYPE> middle_stream;
	hls::stream<Window> window_stream;
	hls::stream<DTYPE> output_stream;

#pragma HLS STREAM variable=pixel_stream depth=3
#pragma HLS STREAM variable=middle_stream depth=655
#pragma HLS STREAM variable=window_stream depth=2000
#pragma HLS STREAM variable=output_stream depth=2000

	ReadFromMem(rows, cols, img_in, pixel_stream, middle_stream);
	Window2D(rows, cols, pixel_stream, window_stream);
	FastCalc(rows, cols, middle_stream, window_stream, threshold, output_stream);
	WriteToMem(rows, cols, output_stream, img_out);


/*
    rows: for (int i=0; i<rows; i++) {
#pragma HLS PIPELINE II=18
    	cols: for (int j=0; j<cols; j++) {
    		if (i < 3 || i > rows-4 || j < 3 || j > cols-4) {
    			img_out[i*cols+j] = 0;
    		}
    		else {
    			DTYPE temp = img_in[i*cols+j];
				int k=0;
    			int c1 = abs(temp - img_in[(i-3)*cols+j]) > threshold;
    			int c2 = abs(temp - img_in[(i+3)*cols+j]) > threshold;
    			int c3 = abs(temp - img_in[i*cols+j-3]) > threshold;
    			int c4 = abs(temp - img_in[i*cols+j+3]) > threshold;
    			int t = c1 + c2 + c3 + c4;
    			if ( t >= 3) {
    				if(abs(temp - img_in[(i-3)*cols+j+1]) > threshold) k++;
    				if(abs(temp - img_in[(i-2)*cols+j+2]) > threshold) k++;
    				if(abs(temp - img_in[(i-1)*cols+j+3]) > threshold) k++;
    				if(abs(temp - img_in[(i+1)*cols+j+3]) > threshold) k++;
    				if(abs(temp - img_in[(i+2)*cols+j+2]) > threshold) k++;
    				if(abs(temp - img_in[(i+3)*cols+j+1]) > threshold) k++;
    				if(abs(temp - img_in[(i+3)*cols+j-1]) > threshold) k++;
    				if(abs(temp - img_in[(i+2)*cols+j-2]) > threshold) k++;
    				if(abs(temp - img_in[(i+1)*cols+j-3]) > threshold) k++;
    				if(abs(temp - img_in[(i-1)*cols+j-3]) > threshold) k++;
    				if(abs(temp - img_in[(i-2)*cols+j-2]) > threshold) k++;
    				if(abs(temp - img_in[(i-3)*cols+j-1]) > threshold) k++;
    				k = k + t;
					if (k >= 12)
						img_out[i*cols+j] = 255;
					else
						img_out[i*cols+j] = 0;
    			}
    			else
    				img_out[i*cols+j] = 0;
    		}
    	}
    }*/
}
