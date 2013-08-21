#include "utils.h"

uint8_t stixel_world::waitForKey()
{
    uint8_t keycode = cv::waitKey(0);
    switch (keycode) {
        case 'q':
            exit(0);
            break;
        default:
            ;
    }
    
    return keycode;
}