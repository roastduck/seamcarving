#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * Proform DP and delete a seam
 * @return : Vector of original coordiates
 */
std::vector<int> dp(cv::Mat3b &image, cv::Mat1i &coord)
{
    assert(coord.empty() || image.rows == coord.rows && image.cols == coord.cols);
    cv::Mat3d dx, dy;
    cv::Sobel(image, dx, CV_64F, 1, 0, 7);
    cv::Sobel(image, dy, CV_64F, 0, 1, 7);
    assert(dx.rows == image.rows && dx.cols == image.cols);
    assert(dy.rows == image.rows && dy.cols == image.cols);
    cv::Mat1d weight(image.rows, image.cols);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            weight(i, j) = norm(dx(i, j)) + norm(dy(i, j));

    cv::Mat1i last(image.rows, image.cols);
    for (int i = 1; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
        {
            double val(weight(i - 1, j)), pos(j);
            if (j > 0 && weight(i - 1, j - 1) < val)
                val = weight(i - 1, j - 1), pos = j - 1;
            if (j < image.cols - 1 && weight(i - 1, j + 1) < val)
                val = weight(i - 1, j + 1), pos = j + 1;
            weight(i, j) += val, last(i, j) = pos;
        }

    std::vector<int> seam(coord.rows); // if coord.empty() than seam.empty()
    cv::Mat3b _image(image.rows, image.cols - 1);
    cv::Mat1i _coord(coord.rows, coord.empty() ? 0 : coord.cols - 1);
    double val(weight(image.rows - 1, 0)), pos(0);
    for (int j = 1; j < image.cols; j++)
        if (weight(image.rows - 1, j) < val)
            val = weight(image.rows - 1, j), pos = j;
    for (int i = image.rows - 1; i >= 0; i--)
    {
        if (!coord.empty())
            seam[i] = coord(i, pos);
        pos = last(i, pos);
        for (int j = 0; j < image.cols - 1; j++)
        {
            _image(i, j) = image(i, j < pos ? j : j + 1);
            if (!coord.empty())
                _coord(i, j) = coord(i, j < pos ? j : j + 1);
        }
    }
    image = _image, coord = _coord;
    return seam;
}

cv::Mat3b carveHoriDec(cv::Mat3b image, int target)
{
    cv::Mat1i coord;
    while (image.cols > target)
        dp(image, coord);
    return image;
}

cv::Mat3b carveHoriInc(cv::Mat3b image, int target)
{
    cv::Mat3b imageDp;
    cv::Mat1i coordDp, coord;
    int origin(0);
    while (image.cols < target)
    {
        if (coord.empty() || imageDp.cols < origin * 0.67)
        {
            coord = cv::Mat1i(image.rows, image.cols);
            for (int i = 0; i < image.rows; i++)
                for (int j = 0; j < image.cols; j++)
                    coord(i, j) = j;
            image.copyTo(imageDp);
            coord.copyTo(coordDp);
            origin = image.cols;
        }
        auto seam = dp(imageDp, coordDp);
        cv::Mat3b _image(image.rows, image.cols + 1);
        cv::Mat1i _coord(coord.rows, coord.cols + 1);
        for (int i = 0; i < image.rows; i++)
        {
            int dec(0);
            for (int j = 0; j < image.cols + 1; j++)
            {
                _image(i, j) = image(i, j - dec), _coord(i, j) = coord(i, j - dec);
                dec |= (coord(i, j) == seam[i]);
            }
        }
        image = _image, coord = _coord;
    }
    return image;
}

/**
 * Seam carving in horizontal direction
 * @param image : The image. This parameter might be varied because of reference counting, but pleaes don't rely on it
 * @param target : Target width
 */
inline cv::Mat3b carveHori(cv::Mat3b image, int target)
{
    return target <= image.cols ? carveHoriDec(image, target) : carveHoriInc(image, target);
}

int main(int argc, const char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage:" << " ./main <input_file> <output_file> <target_width> <target_height>" << std::endl;
        return 1;
    }
    const char *inputFile = argv[1], *outputFile = argv[2];
    int targetWidth, targetHeight;
    sscanf(argv[3], "%d", &targetWidth);
    sscanf(argv[4], "%d", &targetHeight);
    cv::Mat3b image = cv::imread(inputFile);

    image = carveHori(image, targetWidth);
    image = carveHori(image.t(), targetHeight).t();

    assert(image.cols == targetWidth);
    assert(image.rows == targetHeight);
    cv::imwrite(outputFile, image);
    return 0;
}

