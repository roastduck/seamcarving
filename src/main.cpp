#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ui.h"

/**
 * Proform DP and delete a seam
 * @return : Vector of original coordiates
 */
std::vector<int> dp(cv::Mat3b &image, cv::Mat1i &coord, cv::Mat1b &delMask, cv::Mat2i &oriCoord, cv::Mat3b seamsOut)
{
    assert(coord.empty() || image.rows == coord.rows && image.cols == coord.cols);
    assert(image.rows == oriCoord.rows && image.cols == oriCoord.cols);
    assert(delMask.empty() || image.rows == delMask.rows && image.cols == delMask.cols);
    cv::Mat3d dx, dy;
    cv::Sobel(image, dx, CV_64F, 1, 0, 7);
    cv::Sobel(image, dy, CV_64F, 0, 1, 7);
    assert(dx.rows == image.rows && dx.cols == image.cols);
    assert(dy.rows == image.rows && dy.cols == image.cols);
    cv::Mat1d weight(image.rows, image.cols);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            weight(i, j) = !delMask.empty() && delMask(i, j) ? -1e10 : norm(dx(i, j)) + norm(dy(i, j));

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
    cv::Mat2i _oriCoord(oriCoord.rows, oriCoord.cols - 1);
    cv::Mat1b _delMask(delMask.rows, delMask.empty() ? 0 : delMask.cols - 1);
    double val(weight(image.rows - 1, 0)), pos(0);
    for (int j = 1; j < image.cols; j++)
        if (weight(image.rows - 1, j) < val)
            val = weight(image.rows - 1, j), pos = j;
    for (int i = image.rows - 1; i >= 0; i--)
    {
        if (!coord.empty())
            seam[i] = coord(i, pos);
        seamsOut(oriCoord(i, pos)[0], oriCoord(i, pos)[1]) *= 0.2;
        for (int j = 0; j < image.cols - 1; j++)
        {
            _image(i, j) = image(i, j < pos ? j : j + 1);
            if (!coord.empty())
                _coord(i, j) = coord(i, j < pos ? j : j + 1);
            _oriCoord(i, j) = oriCoord(i, j < pos ? j : j + 1);
            if (!delMask.empty())
                _delMask(i, j) = delMask(i, j < pos ? j : j + 1);
        }
        pos = last(i, pos);
    }
    image = _image, coord = _coord, oriCoord = _oriCoord, delMask = _delMask;
    return seam;
}

cv::Mat3b delMasked(cv::Mat3b image, cv::Mat1b delMask, cv::Mat2i &oriCoord, cv::Mat3b seamsOut)
{
    cv::Mat1i coord;
    while (cv::countNonZero(delMask))
        dp(image, coord, delMask, oriCoord, seamsOut);
    return image;
}

cv::Mat3b carveHoriDec(cv::Mat3b image, int target, cv::Mat2i &oriCoord, cv::Mat3b seamsOut)
{
    cv::Mat1i coord;
    cv::Mat1b delMask;
    while (image.cols > target)
        dp(image, coord, delMask, oriCoord, seamsOut);
    return image;
}

cv::Mat3b carveHoriInc(cv::Mat3b image, int target, cv::Mat2i &oriCoord, cv::Mat3b seamsOut)
{
    cv::Mat3b imageDp;
    cv::Mat1i coordDp, coord;
    cv::Mat2i oriCoordDp;
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
            oriCoord.copyTo(oriCoordDp);
            origin = image.cols;
        }
        cv::Mat1b delMask;
        auto seam = dp(imageDp, coordDp, delMask, oriCoordDp, seamsOut);
        cv::Mat3b _image(image.rows, image.cols + 1);
        cv::Mat1i _coord(coord.rows, coord.cols + 1);
        cv::Mat2i _oriCoord(oriCoord.rows, oriCoord.cols + 1);
        for (int i = 0; i < image.rows; i++)
        {
            int dec(0);
            for (int j = 0; j < image.cols + 1; j++)
            {
                _image(i, j) = image(i, j - dec);
                _coord(i, j) = coord(i, j - dec);
                _oriCoord(i, j) = oriCoord(i, j - dec);
                dec |= (coord(i, j) == seam[i]);
            }
        }
        image = _image, coord = _coord, oriCoord = _oriCoord;
    }
    return image;
}

/**
 * Seam carving in horizontal direction
 * @param image : The image. This parameter might be varied because of reference counting, but pleaes don't rely on it
 * @param target : Target width
 */
inline cv::Mat3b carveHori(cv::Mat3b image, int target, cv::Mat2i &oriCoord, cv::Mat3b seamsOut)
{
    return target <= image.cols ? carveHoriDec(image, target, oriCoord, seamsOut) : carveHoriInc(image, target, oriCoord, seamsOut);
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
    cv::Mat2i oriCoord(image.rows, image.cols);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            oriCoord(i, j)[0] = i, oriCoord(i, j)[1] = j;
    cv::Mat3b seamsOut = image.clone();

    cv::Mat1b delMask = getMask(image);

    std::cout << "Processing" << std::endl;
    if (image.cols - targetWidth >= image.rows - targetHeight)
        image = delMasked(image, delMask, oriCoord, seamsOut);
    else
    {
        oriCoord = oriCoord.t();
        image = delMasked(image.t(), delMask.t(), oriCoord, seamsOut).t();
        oriCoord = oriCoord.t();
    }
    image = carveHori(image, targetWidth, oriCoord, seamsOut);
    oriCoord = oriCoord.t();
    image = carveHori(image.t(), targetHeight, oriCoord, seamsOut).t();
    oriCoord = oriCoord.t();

    assert(image.cols == targetWidth);
    assert(image.rows == targetHeight);
    cv::imwrite(outputFile, image);
    cv::imwrite((std::string(outputFile) + ".seams.jpg").c_str(), seamsOut);
    return 0;
}

