#include "test_suite.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;

extern vector<circumscribed_rectangle_coord> detect_faces(Mat image);
extern vector<circumscribed_rectangle_coord> detect_eyes(Mat image, circumscribed_rectangle_coord face);
extern Mat detect_and_correct_red_eyes(Mat image, vector<circumscribed_rectangle_coord> eyes);

void SimpleRedEyeTester::run_all_tests() {
    cout << "=== RED-EYE CORRECTION TEST SUITE ===" << endl << endl;

    // Test 1: Normal case - single person with red-eye
    test_single_person();

    // Test 2: Group photo
    test_group_photo();

    // Test 3: No red-eye (should not over-correct)
    test_no_red_eye();

    // Test 4: No face in image
    test_no_face();

    // Test 5: Face but eyes closed/not visible
    test_closed_eyes();

    print_summary();
}

void SimpleRedEyeTester::test_single_person() {
    cout << "TEST 1: Single Person with Red-Eye" << endl;

    auto start = chrono::high_resolution_clock::now();

    Mat image = imread("../test_images/single_redeye.jpg");
    if (image.empty()) {
        results.push_back({"Single Person", false, 0, 0, 0, "Image not found"});
        cout << " FAILED: Image not found" << endl << endl;
        return;
    }

    // Run detection
    vector<circumscribed_rectangle_coord> faces = detect_faces(image);
    vector<circumscribed_rectangle_coord> all_eyes;

    for (size_t i = 0; i < faces.size(); i++) {
        vector<circumscribed_rectangle_coord> eyes = detect_eyes(image, faces[i]);
        all_eyes.insert(all_eyes.end(), eyes.begin(), eyes.end());
    }

    Mat corrected = detect_and_correct_red_eyes(image, all_eyes);

    auto end = chrono::high_resolution_clock::now();
    double time_ms = chrono::duration<double, milli>(end - start).count();

    // Evaluate results
    bool success = (faces.size() == 1) && (all_eyes.size() == 2);
    string notes = "Expected: 1 face, 2 eyes. Got: " + to_string(faces.size()) +
                   " faces, " + to_string(all_eyes.size()) + " eyes";

    results.push_back({"Single Person", success, (int)faces.size(), (int)all_eyes.size(), time_ms, notes});

    if (success) {
        cout << " PASSED: " << notes << endl;
        imwrite("../test_results/single_corrected.jpg", corrected);
    } else {
        cout << " FAILED: " << notes << endl;
    }
    cout << "Processing time: " << time_ms << " ms" << endl << endl;
}

void SimpleRedEyeTester::test_group_photo() {
    cout << "TEST 2: Group Photo (2 people)" << endl;

    auto start = chrono::high_resolution_clock::now();

    Mat image = imread("../test_images/group_redeye.jpg");
    if (image.empty()) {
        results.push_back({"Group Photo", false, 0, 0, 0, "Image not found"});
        cout << " FAILED: Image not found" << endl << endl;
        return;
    }

    vector<circumscribed_rectangle_coord> faces = detect_faces(image);
    vector<circumscribed_rectangle_coord> all_eyes;

    for (size_t i = 0; i < faces.size(); i++) {
        vector<circumscribed_rectangle_coord> eyes = detect_eyes(image, faces[i]);
        all_eyes.insert(all_eyes.end(), eyes.begin(), eyes.end());
    }

    Mat corrected = detect_and_correct_red_eyes(image, all_eyes);

    auto end = chrono::high_resolution_clock::now();
    double time_ms = chrono::duration<double, milli>(end - start).count();

    bool success = (faces.size() >= 2) && (all_eyes.size() >= 4);
    string notes = "Expected: ≥2 faces, ≥4 eyes. Got: " + to_string(faces.size()) +
                   " faces, " + to_string(all_eyes.size()) + " eyes";

    results.push_back({"Group Photo", success, (int)faces.size(), (int)all_eyes.size(), time_ms, notes});

    if (success) {
        cout << " PASSED: " << notes << endl;
        imwrite("../test_results/group_corrected.jpg", corrected);
    } else {
        cout << " FAILED: " << notes << endl;
    }
    cout << "Processing time: " << time_ms << " ms" << endl << endl;
}

void SimpleRedEyeTester::test_no_red_eye() {
    cout << "TEST 3: Normal Photo (No Red-Eye)" << endl;

    auto start = chrono::high_resolution_clock::now();

    Mat image = imread("../test_images/normal_photo.jpg");
    if (image.empty()) {
        results.push_back({"No Red-Eye", false, 0, 0, 0, "Image not found"});
        cout << " FAILED: Image not found" << endl << endl;
        return;
    }

    vector<circumscribed_rectangle_coord> faces = detect_faces(image);
    vector<circumscribed_rectangle_coord> all_eyes;

    for (size_t i = 0; i < faces.size(); i++) {
        vector<circumscribed_rectangle_coord> eyes = detect_eyes(image, faces[i]);
        all_eyes.insert(all_eyes.end(), eyes.begin(), eyes.end());
    }

    Mat corrected = detect_and_correct_red_eyes(image, all_eyes);

    auto end = chrono::high_resolution_clock::now();
    double time_ms = chrono::duration<double, milli>(end - start).count();

    bool success = faces.size() >= 1;
    string notes = "Should detect faces without over-correcting. Found: " +
                   to_string(faces.size()) + " faces, " + to_string(all_eyes.size()) + " eyes";

    results.push_back({"No Red-Eye", success, (int)faces.size(), (int)all_eyes.size(), time_ms, notes});

    if (success) {
        cout << " PASSED: " << notes << endl;
        imwrite("../test_results/normal_processed.jpg", corrected);
    } else {
        cout << " FAILED: " << notes << endl;
    }
    cout << "Processing time: " << time_ms << " ms" << endl << endl;
}

void SimpleRedEyeTester::test_no_face() {
    cout << "TEST 4: No Face in Image" << endl;

    auto start = chrono::high_resolution_clock::now();

    Mat image = imread("../test_images/landscape.jpg");
    if (image.empty()) {
        results.push_back({"No Face", false, 0, 0, 0, "Image not found"});
        cout << " FAILED: Image not found" << endl << endl;
        return;
    }

    vector<circumscribed_rectangle_coord> faces = detect_faces(image);

    auto end = chrono::high_resolution_clock::now();
    double time_ms = chrono::duration<double, milli>(end - start).count();

    bool success = faces.size() == 0;
    string notes = "Expected: 0 faces. Got: " + to_string(faces.size()) + " faces";

    results.push_back({"No Face", success, (int)faces.size(), 0, time_ms, notes});

    if (success) {
        cout << " PASSED: " << notes << endl;
    } else {
        cout << " FAILED: " << notes << " (False positive detection)" << endl;
    }
    cout << "Processing time: " << time_ms << " ms" << endl << endl;
}

void SimpleRedEyeTester::test_closed_eyes() {
    cout << "TEST 5: Face with Closed Eyes" << endl;

    auto start = chrono::high_resolution_clock::now();

    Mat image = imread("../test_images/closed_eyes.jpg");
    if (image.empty()) {
        results.push_back({"Closed Eyes", false, 0, 0, 0, "Image not found"});
        cout << " FAILED: Image not found" << endl << endl;
        return;
    }

    vector<circumscribed_rectangle_coord> faces = detect_faces(image);
    vector<circumscribed_rectangle_coord> all_eyes;

    for (size_t i = 0; i < faces.size(); i++) {
        vector<circumscribed_rectangle_coord> eyes = detect_eyes(image, faces[i]);
        all_eyes.insert(all_eyes.end(), eyes.begin(), eyes.end());
    }

    auto end = chrono::high_resolution_clock::now();
    double time_ms = chrono::duration<double, milli>(end - start).count();

    bool success = (faces.size() >= 1) && (all_eyes.size() < 2);
    string notes = "Expected: ≥1 face, <2 eyes. Got: " + to_string(faces.size()) +
                   " faces, " + to_string(all_eyes.size()) + " eyes";

    results.push_back({"Closed Eyes", success, (int)faces.size(), (int)all_eyes.size(), time_ms, notes});

    if (success) {
        cout << " PASSED: " << notes << endl;
    } else {
        cout << " FAILED: " << notes << endl;
    }
    cout << "Processing time: " << time_ms << " ms" << endl << endl;
}

void SimpleRedEyeTester::print_summary() {
    cout << "=== TEST SUMMARY ===" << endl;
    int passed = 0, total = results.size();
    double avg_time = 0;

    for (const auto& result : results) {
        if (result.success) passed++;
        avg_time += result.processing_time;

        cout << result.test_name << ": "
             << (result.success ? "" : "") << " "
             << result.notes << endl;
    }

    avg_time /= total;

    cout << endl;
    cout << "Overall: " << passed << "/" << total << " tests passed" << endl;
    cout << "Average processing time: " << avg_time << " ms" << endl;
    cout << "Success rate: " << (100.0 * passed / total) << "%" << endl;
}