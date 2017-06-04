#include <iostream>
#include <algorithm>
#include "ui.h"

struct WindowParam
{
    cv::Mat3b img;
    cv::Mat1b mask; /// Using reference counting, no need to be reference
};

void onMonse(int event, int x, int y, int flags, void *_param)
{
    WindowParam *param = (WindowParam*)_param;
    if ((event == CV_EVENT_MOUSEMOVE || event == CV_EVENT_LBUTTONDOWN) && (flags & CV_EVENT_FLAG_LBUTTON))
    {
        int radius = 10;
        for (int i = std::max(0, y - radius); i <= std::min(param->img.rows - 1, y + radius); i++)
            for (int j = std::max(0, x - radius); j <= std::min(param->img.cols - 1, x + radius); j++)
                if (cv::norm(cv::Point(j, i) - cv::Point(x, y)) <= radius && !param->mask(i, j))
                {
                    param->img(i, j) = param->img(i, j) * 0.7 + cv::Vec3b(0, 255, 0) * 0.3; // BGR
                    param->mask(i, j) = 1;
                }
    }
}

cv::Mat1b getMask(const cv::Mat3b img)
{
    std::cout << "Please use the cursor to mark the area(s) you want to delete" << std::endl;
    std::cout << "Press <ESC> to finish" << std::endl;

    cv::namedWindow("Draw Mask Area", CV_WINDOW_AUTOSIZE);
    cv::Mat1b mask(img.rows, img.cols, (unsigned char)0);
    WindowParam param({img.clone(), mask});
    cv::setMouseCallback("Draw Mask Area", onMonse, (void*)&param);
    do cv::imshow("Draw Mask Area", param.img); while (cv::waitKey(100) != 27);
    cv::setMouseCallback("Draw Mask Area", 0, 0);
    cv::destroyAllWindows();
    return mask;
}

