#include "utils.h"

uint8_t stixel_world::waitForKey(uint32_t * time)
{   
    uint8_t keycode;
    if (time == NULL)
        keycode = cv::waitKey(0);
    else
        keycode = cv::waitKey(*time);
    
    switch (keycode) {
        case 'q':
        case 0x1B:  // ESC
            exit(0);
            break;
        case 0x20:  // SPACE
            *time = 0;
            break;
        case 0x43:  // c
        case 0x63:  // C
            *time = 20;
            break;
            
        default:
            ;
    }
    
    return keycode;
}