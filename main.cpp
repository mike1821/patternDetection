#include <stdio.h>
#include <iostream>
#include <patternDetection.h>
#include <definitions.h>
#include "Config.h"

patternDetection pD;

//Activated a GIT learning session based on this file
const int TICKET_NUM=10;

int main()
{
    //working with opencv images
    Mat input;
    char tmp[256];

    for(int i=0; i<TICKET_NUM;i++){
        //sprintf(tmp,"C:\\temp\\MTimage-1444045%03d.jpg",i);
        sprintf(tmp,"C:\\temp\\scan_up01.bmp");
        input = imread(tmp);
        std::cout << "Feeding image: " << tmp << std::endl;
        pD.processCoupon(input, POWERBALL_PLAYSLIP_ID);
    }

    return 0;
}

