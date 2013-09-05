#include "utils.h"

uint8_t stixel_world::waitForKey(uint32_t time)
{
    uint8_t keycode = cv::waitKey(20);
    switch (keycode) {
        case 'q':
            exit(0);
            break;
        default:
            ;
    }
    
    return keycode;
}