
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fast.h"

#include <iostream>
using namespace std;

int main()
{
    DTYPE input[16384];
    DTYPE output[16384];
    int gold;
    int in;

    int imgheight = 128;
    int imgwidth = 128;

    int threshold = 20;

    FILE * fp = fopen("C:\\Users\\Kuo\\Vitis_HLS\\AAHLS_LabC_FAST\\source_code\\src.txt","r");
    FILE * fpo = fopen("C:\\Users\\Kuo\\Vitis_HLS\\AAHLS_LabC_FAST\\source_code\\dst.txt","r");

    for(int i=0; i<16384; i++)
    {
        fscanf(fp, "%d", &in);
        input[i] = in;
    }

	fast_accel(input, threshold, output, imgheight, imgwidth);
    
    int tf = 0;

    for(int i=0; i<16384; i++)
    {
        fscanf(fpo, "%d", &gold);
        //cout << output[i] << " " << gold << endl;
        if (output[i] - gold != 0)
        {
        	cout << i << " " << output[i] << " " << gold << endl;
            tf = 1;
        }        
    }

    fclose(fp);
    fclose(fpo);

    if (tf == 1)
    {
        fprintf(stdout, "*******************************************\n");
        fprintf(stdout, "FAIL: Output DOES NOT match the golden output\n");
        fprintf(stdout, "*******************************************\n");
        return 0;
    } 
    else 
    {
        fprintf(stdout, "*******************************************\n");
        fprintf(stdout, "PASS: The output matches the golden output!\n");
        fprintf(stdout, "*******************************************\n");
        return 0;
    }

}
