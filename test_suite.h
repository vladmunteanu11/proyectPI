#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

struct circumscribed_rectangle_coord {
    int r_min;
    int r_max;
    int c_min;
    int c_max;
};

struct TestResult {
    string test_name;
    bool success;
    int faces_found;
    int eyes_found;
    double processing_time;
    string notes;
};

class SimpleRedEyeTester {
private:
    vector<TestResult> results;

    void test_single_person();
    void test_group_photo();
    void test_no_red_eye();
    void test_no_face();
    void test_closed_eyes();
    void print_summary();

public:
    void run_all_tests();
};

#endif // TEST_SUITE_H