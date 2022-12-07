#ifndef DEFINITIONS
#define DEFINITIONS

#define DEBUG
#define DEBUG_BETBOXES

const int PIXEL2MM = 0.471;

//logo area limits
#define AREA_MIN_SIZE 200000
#define AREA_MAX_SIZE 1400000 //600000
#define AREA_THRESHOLD 90000

#define GRID_SIZE 200000

//Area limits
#define BOX_MIN_SIZE 550//250
#define BOX_MAX_SIZE 1100//550


const int TOP_LEFT     = 0;
const int TOP_RIGHT    = 1;
const int BOTTOM_RIGHT = 2;
const int BOTTOM_LEFT  = 3;

const int FRAME_WIDTH   = 1693;//1280;
const int FRAME_HEIGHT  = 862;//720;
const int CROP_EXTEND_X = 15;
const int CROP_EXTEND_Y = 6;

//Number of cells of the given coupon
const int POWERBALL_NUMBER_OF_BOXES = 390;
const int KENO_NUMBER_OF_BOXES      = 417;

#endif // DEFINITIONS


