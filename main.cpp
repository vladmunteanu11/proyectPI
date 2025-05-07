#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Function to check if a point is inside an image
bool IsInside(Mat img, int i, int j) {
    return i >= 0 && j >= 0 && i < img.rows && j < img.cols;
}

// Structure for BGR channels
struct image_channels_bgr {
    Mat B, G, R;
};

// Breaking the image into BGR channels
image_channels_bgr break_channels(Mat source) {
    int rows = source.rows;
    int cols = source.cols;

    Mat B = Mat(rows, cols, CV_8UC1);
    Mat G = Mat(rows, cols, CV_8UC1);
    Mat R = Mat(rows, cols, CV_8UC1);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            B.at<uchar>(i, j) = source.at<Vec3b>(i, j)[0];
            G.at<uchar>(i, j) = source.at<Vec3b>(i, j)[1];
            R.at<uchar>(i, j) = source.at<Vec3b>(i, j)[2];
        }
    }

    image_channels_bgr bgr_channels;
    bgr_channels.B = B;
    bgr_channels.G = G;
    bgr_channels.R = R;

    return bgr_channels;
}

// Structure for HSV channels
struct image_channels_hsv {
    Mat H, S, V;
};

// Converting BGR to HSV
image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels) {
    int rows = bgr_channels.R.rows;
    int cols = bgr_channels.R.cols;

    Mat H = Mat(rows, cols, CV_32F);
    Mat S = Mat(rows, cols, CV_32F);
    Mat V = Mat(rows, cols, CV_32F);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            float r = (float)bgr_channels.R.at<uchar>(i, j) / 255.0f;
            float g = (float)bgr_channels.G.at<uchar>(i, j) / 255.0f;
            float b = (float)bgr_channels.B.at<uchar>(i, j) / 255.0f;

            float M = max(r, max(g, b));
            float m = min(r, min(g, b));
            float C = M - m;

            V.at<float>(i, j) = M;
            S.at<float>(i, j) = (M != 0) ? (C / M) : 0.0f;

            if(C != 0) {
                if(M == r)
                    H.at<float>(i, j) = 60.0f * ((g - b) / C);
                else if(M == g)
                    H.at<float>(i, j) = 120.0f + 60.0f * ((b - r) / C);
                else
                    H.at<float>(i, j) = 240.0f + 60.0f * ((r - g) / C);
            } else {
                H.at<float>(i, j) = 0.0f;
            }

            if(H.at<float>(i, j) < 0)
                H.at<float>(i, j) += 360.0f;
        }
    }

    image_channels_hsv hsv_channels;
    hsv_channels.H = H;
    hsv_channels.S = S;
    hsv_channels.V = V;

    return hsv_channels;
}

// Structure for morphological operations
struct neighborhood_structure {
    int* di;
    int* dj;
    int size;
};

// Dilation operation
Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat dst = Mat::ones(source.rows, source.cols, CV_8UC1) * 255;
    Mat aux = source.clone();

    for(int nr = 0; nr < no_iter; nr++) {
        for(int i = 0; i < source.rows; i++) {
            for(int j = 0; j < source.cols; j++) {
                if(aux.at<uchar>(i, j) == 0) {
                    for(int k = 0; k < neighborhood.size; k++) {
                        int nx = j + neighborhood.dj[k];
                        int ny = i + neighborhood.di[k];
                        if(IsInside(source, ny, nx))
                            dst.at<uchar>(ny, nx) = 0;
                    }
                }
            }
        }
        aux = dst.clone();
    }

    return dst;
}

// Erosion operation
Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat dst = Mat::ones(source.rows, source.cols, CV_8UC1) * 255;
    Mat aux = source.clone();

    for(int nr = 0; nr < no_iter; nr++) {
        for(int i = 0; i < source.rows; i++) {
            for(int j = 0; j < source.cols; j++) {
                if(aux.at<uchar>(i, j) == 0) {
                    bool only_obj = true;
                    for(int k = 0; k < neighborhood.size; k++) {
                        int nx = j + neighborhood.dj[k];
                        int ny = i + neighborhood.di[k];
                        if(IsInside(source, ny, nx))
                            if(aux.at<uchar>(ny, nx) == 255)
                                only_obj = false;
                    }
                    if(only_obj) {
                        dst.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
        aux = dst.clone();
    }

    return dst;
}

// Opening operation
Mat opening(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat aux = source.clone();
    for(int nr = 0; nr < no_iter; nr++) {
        aux = dilation(erosion(aux, neighborhood, 1), neighborhood, 1);
    }
    return aux;
}

// Closing operation
Mat closing(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat aux = source.clone();
    for(int nr = 0; nr < no_iter; nr++) {
        aux = erosion(dilation(aux, neighborhood, 1), neighborhood, 1);
    }
    return aux;
}

// Structure for connected component labeling
struct labels {
    Mat labels;
    int no_labels;
};

// Connected component labeling using BFS
labels BFS_labeling(Mat source) {
    int rows = source.rows;
    int cols = source.cols;
    Mat labels_mat = Mat::zeros(rows, cols, CV_32SC1);
    int no_labels = 0;

    // Define 8-neighborhood
    int n8_di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int n8_dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(source.at<uchar>(i, j) == 0 && labels_mat.at<int>(i, j) == 0) {
                no_labels++;
                queue<Point> q;
                labels_mat.at<int>(i, j) = no_labels;
                q.push(Point(j, i));

                while(!q.empty()) {
                    Point curr = q.front();
                    q.pop();

                    for(int k = 0; k < 8; k++) {
                        int ny = curr.y + n8_di[k];
                        int nx = curr.x + n8_dj[k];

                        if(IsInside(source, ny, nx) && source.at<uchar>(ny, nx) == 0 && labels_mat.at<int>(ny, nx) == 0) {
                            labels_mat.at<int>(ny, nx) = no_labels;
                            q.push(Point(nx, ny));
                        }
                    }
                }
            }
        }
    }

    labels result = {labels_mat, no_labels};
    return result;
}

// Structure for bounding rectangle
struct circumscribed_rectangle_coord {
    int r_min;
    int r_max;
    int c_min;
    int c_max;
};

// Compute bounding rectangles for each component
vector<circumscribed_rectangle_coord> compute_bounding_boxes(Mat binary_object, labels component_labels) {
    vector<circumscribed_rectangle_coord> boxes(component_labels.no_labels + 1);

    // Initialize values
    for(int i = 0; i <= component_labels.no_labels; i++) {
        boxes[i].r_min = binary_object.rows;
        boxes[i].r_max = 0;
        boxes[i].c_min = binary_object.cols;
        boxes[i].c_max = 0;
    }

    // Find min/max coordinates for each component
    for(int i = 0; i < binary_object.rows; i++) {
        for(int j = 0; j < binary_object.cols; j++) {
            if(binary_object.at<uchar>(i, j) == 0) {
                int label = component_labels.labels.at<int>(i, j);

                // Update bounding box
                if(i < boxes[label].r_min) boxes[label].r_min = i;
                if(i > boxes[label].r_max) boxes[label].r_max = i;
                if(j < boxes[label].c_min) boxes[label].c_min = j;
                if(j > boxes[label].c_max) boxes[label].c_max = j;
            }
        }
    }

    return boxes;
}

// Calculate aspect ratio (width/height) of a rectangle
float compute_aspect_ratio(circumscribed_rectangle_coord coord) {
    return (float)(coord.c_max - coord.c_min + 1) / (float)(coord.r_max - coord.r_min + 1);
}

// Compute area of each component
vector<int> compute_areas(Mat binary_object, labels component_labels) {
    vector<int> areas(component_labels.no_labels + 1, 0);

    for(int i = 0; i < binary_object.rows; i++) {
        for(int j = 0; j < binary_object.cols; j++) {
            if(binary_object.at<uchar>(i, j) == 0) {
                int label = component_labels.labels.at<int>(i, j);
                areas[label]++;
            }
        }
    }

    return areas;
}

// Create a mask focusing on the central region where faces typically appear
Mat preprocess_for_face_detection(Mat image) {
    int rows = image.rows;
    int cols = image.cols;

    // Create a mask focusing on the central portion of the image
    Mat center_mask = Mat(rows, cols, CV_8UC1, Scalar(255));

    // Define the central region (center 70% of the image)
    int center_width = cols * 0.7;
    int center_height = rows * 0.7;
    int start_x = (cols - center_width) / 2;
    int start_y = (rows - center_height) / 2;

    // Mark the central region
    for(int i = start_y; i < start_y + center_height; i++) {
        for(int j = start_x; j < start_x + center_width; j++) {
            center_mask.at<uchar>(i, j) = 0; // Mark central area
        }
    }

    return center_mask;
}

// Improved skin detection with stricter parameters
Mat detect_skin(image_channels_hsv hsv_channels, image_channels_bgr bgr_channels) {
    int rows = hsv_channels.H.rows;
    int cols = hsv_channels.H.cols;

    Mat skin_mask = Mat(rows, cols, CV_8UC1, Scalar(255));

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            float h = hsv_channels.H.at<float>(i, j);
            float s = hsv_channels.S.at<float>(i, j);
            float v = hsv_channels.V.at<float>(i, j);

            // Get RGB values for additional checks
            int r = bgr_channels.R.at<uchar>(i, j);
            int g = bgr_channels.G.at<uchar>(i, j);
            int b = bgr_channels.B.at<uchar>(i, j);

            // Stricter skin tone criteria for fair-skinned children
            bool is_skin_hue = (h >= 0 && h <= 25) || (h >= 335 && h <= 360);
            bool is_proper_saturation = (s >= 0.1f && s <= 0.6f);
            bool is_proper_value = (v >= 0.4f && v <= 0.9f);

            // Stricter RGB check for better skin detection
            bool rgb_check = (r > 95) && (g > 40) && (b > 20) &&
                             (r > g) && (r > b) &&
                             (abs(r - g) > 15);  // Require stronger red component

            if(is_skin_hue && is_proper_saturation && is_proper_value && rgb_check) {
                skin_mask.at<uchar>(i, j) = 0;  // Mark as skin
            }
        }
    }

    return skin_mask;
}

// Improved face detection that avoids selecting the whole image
vector<circumscribed_rectangle_coord> detect_faces(Mat image) {
    // Break image into channels
    image_channels_bgr bgr_channels = break_channels(image);
    image_channels_hsv hsv_channels = bgr_2_hsv(bgr_channels);

    // First, focus on the central region of the image
    Mat center_mask = preprocess_for_face_detection(image);

    // Detect skin with stricter parameters
    Mat skin_mask = detect_skin(hsv_channels, bgr_channels);

    // Combine with center mask to focus only on central region
    for(int i = 0; i < skin_mask.rows; i++) {
        for(int j = 0; j < skin_mask.cols; j++) {
            if(center_mask.at<uchar>(i, j) == 255) { // If not in center region
                skin_mask.at<uchar>(i, j) = 255;     // Set to background
            }
        }
    }

    // Setup morphological operations
    int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    neighborhood_structure n8 = {di, dj, 8};

    // More controlled morphological operations
    Mat cleaned_mask = opening(skin_mask, n8, 1);    // Remove noise
    cleaned_mask = closing(cleaned_mask, n8, 2);     // Less aggressive closing
    cleaned_mask = opening(cleaned_mask, n8, 1);     // Clean edges

    // Find connected components
    labels component_labels = BFS_labeling(cleaned_mask);
    vector<int> areas = compute_areas(cleaned_mask, component_labels);
    vector<circumscribed_rectangle_coord> boxes = compute_bounding_boxes(cleaned_mask, component_labels);

    // Advanced face candidate selection
    vector<circumscribed_rectangle_coord> face_candidates;
    int total_area = image.rows * image.cols;

    // Sort components by area (descending)
    vector<pair<int, int>> area_indices;
    for(int i = 1; i <= component_labels.no_labels; i++) {
        area_indices.push_back({areas[i], i});
    }
    sort(area_indices.begin(), area_indices.end(), greater<pair<int, int>>());

    // Take largest component that meets stricter face criteria
    for(auto& p : area_indices) {
        int area = p.first;
        int idx = p.second;

        // Face must be significant but not too large (5-60% of total image)
        bool is_proper_size = (area > total_area * 0.05) && (area < total_area * 0.6);

        // Face aspect ratio check (width/height)
        float aspect_ratio = compute_aspect_ratio(boxes[idx]);
        bool has_face_aspect_ratio = (aspect_ratio >= 0.7 && aspect_ratio <= 1.3);  // Stricter

        // Additional check: Face should be reasonably centered
        int center_x = (boxes[idx].c_min + boxes[idx].c_max) / 2;
        int center_y = (boxes[idx].r_min + boxes[idx].r_max) / 2;
        bool is_centered = (center_x > image.cols * 0.25 && center_x < image.cols * 0.75 &&
                            center_y > image.rows * 0.25 && center_y < image.rows * 0.75);

        if(is_proper_size && has_face_aspect_ratio && is_centered) {
            face_candidates.push_back(boxes[idx]);
            break;  // Take only the largest valid component
        }
    }

    return face_candidates;
}

// Eye detection within a face region using gradient and color information
vector<circumscribed_rectangle_coord> detect_eyes(Mat image, circumscribed_rectangle_coord face) {
    // Focus specifically on the eye region within the face
    int face_height = face.r_max - face.r_min + 1;
    int face_width = face.c_max - face.c_min + 1;

    // Define a more precise eye region (upper 30% of face)
    int eye_region_height = face_height * 0.3;
    int r_min = face.r_min + face_height * 0.18;  // Start ~18% from top of face
    int r_max = r_min + eye_region_height;

    // Narrow horizontally to focus on eye area (avoid sides of face)
    int width_margin = face_width * 0.1;
    int c_min = face.c_min + width_margin;
    int c_max = face.c_max - width_margin;

    // Extract the eye region
    Mat eye_roi = image(Range(r_min, r_max + 1), Range(c_min, c_max + 1)).clone();

    // Convert to grayscale for better feature detection
    Mat grayscale = Mat(eye_roi.rows, eye_roi.cols, CV_8UC1);
    for(int i = 0; i < eye_roi.rows; i++) {
        for(int j = 0; j < eye_roi.cols; j++) {
            grayscale.at<uchar>(i, j) = (eye_roi.at<Vec3b>(i, j)[0] +
                                         eye_roi.at<Vec3b>(i, j)[1] +
                                         eye_roi.at<Vec3b>(i, j)[2]) / 3;
        }
    }

    // Apply histogram equalization for better contrast
    int* histogram = (int*)calloc(256, sizeof(int));
    for(int i = 0; i < grayscale.rows; i++) {
        for(int j = 0; j < grayscale.cols; j++) {
            histogram[grayscale.at<uchar>(i, j)]++;
        }
    }

    float* pdf = (float*)calloc(256, sizeof(float));
    float* cpdf = (float*)calloc(256, sizeof(float));
    int total_pixels = grayscale.rows * grayscale.cols;

    for(int i = 0; i < 256; i++) {
        pdf[i] = (float)histogram[i] / (float)total_pixels;
    }

    cpdf[0] = pdf[0];
    for(int i = 1; i < 256; i++) {
        cpdf[i] = cpdf[i-1] + pdf[i];
    }

    Mat equalized = Mat(grayscale.rows, grayscale.cols, CV_8UC1);
    for(int i = 0; i < grayscale.rows; i++) {
        for(int j = 0; j < grayscale.cols; j++) {
            equalized.at<uchar>(i, j) = 255 * cpdf[grayscale.at<uchar>(i, j)];
        }
    }

    // Create a binary map for eye candidates
    Mat eye_candidates = Mat(equalized.rows, equalized.cols, CV_8UC1, Scalar(255));

    // Extract color information for analysis
    image_channels_bgr bgr_channels = break_channels(eye_roi);
    image_channels_hsv hsv_channels = bgr_2_hsv(bgr_channels);

    // Enhanced eye detection focusing on pupil and iris
    for(int i = 0; i < eye_roi.rows; i++) {
        for(int j = 0; j < eye_roi.cols; j++) {
            // Color analysis
            Vec3b pixel = eye_roi.at<Vec3b>(i, j);
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];

            float h = hsv_channels.H.at<float>(i, j);
            float s = hsv_channels.S.at<float>(i, j);
            float v = hsv_channels.V.at<float>(i, j);

            // More specific pupil detection (focus on dark centers)
            bool is_pupil = (v < 0.45f) && (equalized.at<uchar>(i, j) < 100);

            // More specific iris detection
            bool is_iris = (h >= 180 && h <= 250) && (s > 0.2f && s < 0.7f);

            // Red-eye detection
            bool is_red_eye = (r > 120) && (r > b * 1.4) && (r > g * 1.4);

            // Focus more on core eye features and less on surrounding area
            if(is_pupil || is_red_eye || is_iris) {
                eye_candidates.at<uchar>(i, j) = 0;
            }
        }
    }

    // Apply morphological operations
    int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    neighborhood_structure n8 = {di, dj, 8};

    // First clean up noise
    Mat cleaned = opening(eye_candidates, n8, 1);

    // Then connect nearby eye pixels but with less dilation to keep rectangles small
    cleaned = closing(cleaned, n8, 1);

    // Find connected components
    labels component_labels = BFS_labeling(cleaned);
    vector<int> areas = compute_areas(cleaned, component_labels);
    vector<circumscribed_rectangle_coord> boxes = compute_bounding_boxes(cleaned, component_labels);

    // Filter eye candidates - use tighter criteria for smaller boxes
    vector<circumscribed_rectangle_coord> eye_regions;

    // Expected eye size (smaller to get tighter boxes)
    float expected_eye_width = face_width * 0.1;  // ~10% of face width
    float expected_eye_height = face_height * 0.05; // ~5% of face height

    for(int i = 1; i <= component_labels.no_labels; i++) {
        int comp_width = boxes[i].c_max - boxes[i].c_min + 1;
        int comp_height = boxes[i].r_max - boxes[i].r_min + 1;

        // More restrictive size check for tighter boxes
        bool proper_size = (comp_width >= expected_eye_width * 0.4f) &&
                           (comp_width <= expected_eye_width * 2.0f) &&
                           (comp_height >= expected_eye_height * 0.4f) &&
                           (comp_height <= expected_eye_height * 2.0f);

        // Check shape - eyes are roughly circular/oval
        float aspect_ratio = (float)comp_width / (float)comp_height;
        bool proper_shape = (aspect_ratio >= 0.7f && aspect_ratio <= 2.0f);

        // Area check - not too small to miss eyes, not too large to include more than the eye
        int min_area = 30;  // Minimum pixels
        int max_area = face_width * face_height * 0.01; // Max ~1% of face area (smaller than before)
        bool proper_area = (areas[i] > min_area && areas[i] < max_area);

        if(proper_size && proper_shape && proper_area) {
            // Convert to original image coordinates
            circumscribed_rectangle_coord eye_in_original;
            eye_in_original.r_min = boxes[i].r_min + r_min;
            eye_in_original.r_max = boxes[i].r_max + r_min;
            eye_in_original.c_min = boxes[i].c_min + c_min;
            eye_in_original.c_max = boxes[i].c_max + c_min;

            eye_regions.push_back(eye_in_original);
        }
    }

    // Free memory
    free(histogram);
    free(pdf);
    free(cpdf);

    // If we find more than 2 candidates, select a left-right pair
    if(eye_regions.size() > 2) {
        // Sort by x-coordinate
        sort(eye_regions.begin(), eye_regions.end(),
             [](const circumscribed_rectangle_coord& a, const circumscribed_rectangle_coord& b) {
                 return a.c_min < b.c_min;
             });

        // Find the left and right eye with appropriate spacing
        vector<circumscribed_rectangle_coord> best_pair;

        // This should be the left eye
        best_pair.push_back(eye_regions.front());

        // Find right eye candidate at appropriate distance
        int face_width = face.c_max - face.c_min + 1;
        int left_eye_x = (eye_regions.front().c_min + eye_regions.front().c_max) / 2;

        for(int i = eye_regions.size() - 1; i > 0; i--) {
            int current_x = (eye_regions[i].c_min + eye_regions[i].c_max) / 2;
            int distance = current_x - left_eye_x;

            // Check if this is a reasonable distance for the right eye
            if(distance > face_width * 0.25 && distance < face_width * 0.7) {
                best_pair.push_back(eye_regions[i]);
                return best_pair;
            }
        }

        // If no suitable right eye, just take rightmost
        if(best_pair.size() == 1 && eye_regions.size() > 1) {
            best_pair.push_back(eye_regions.back());
        }

        return best_pair;
    }

    return eye_regions;
}

// Red-eye detection and correction
Mat detect_and_correct_red_eyes(Mat image, vector<circumscribed_rectangle_coord> eyes) {
    Mat result = image.clone();

    // Process each eye
    for(size_t i = 0; i < eyes.size(); i++) {
        // Extract the eye region with slight padding
        int padding = 2;
        int r_min = max(0, eyes[i].r_min - padding);
        int r_max = min(image.rows - 1, eyes[i].r_max + padding);
        int c_min = max(0, eyes[i].c_min - padding);
        int c_max = min(image.cols - 1, eyes[i].c_max + padding);

        // Extract the eye region
        Mat eye_region = image(Range(r_min, r_max + 1), Range(c_min, c_max + 1)).clone();

        // Convert to HSV for better red detection
        image_channels_bgr bgr_channels = break_channels(eye_region);
        image_channels_hsv hsv_channels = bgr_2_hsv(bgr_channels);

        // Create a mask for this eye's red pixels
        Mat red_mask = Mat(eye_region.rows, eye_region.cols, CV_8UC1, Scalar(255));

        // Find the average eye color (for surrounding iris)
        Vec3f avg_eye_color(0, 0, 0);
        int non_red_pixels = 0;

        // First pass: Identify red pixels and compute average non-red eye color
        for(int r = 0; r < eye_region.rows; r++) {
            for(int c = 0; c < eye_region.cols; c++) {
                Vec3b pixel = eye_region.at<Vec3b>(r, c);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                float h = hsv_channels.H.at<float>(r, c);
                float s = hsv_channels.S.at<float>(r, c);
                float v = hsv_channels.V.at<float>(r, c);

                // Enhanced red-eye detection
                bool is_red_eye = (
                        // Traditional RGB check
                        (red > 120 && red > blue * 1.4 && red > green * 1.4) ||
                        // HSV check for red hues
                        ((h < 20 || h > 340) && s > 0.5f && v > 0.4f)
                );

                if(is_red_eye) {
                    red_mask.at<uchar>(r, c) = 0;  // Mark as red-eye
                } else {
                    // Only use moderately bright pixels for average (avoid very dark or bright areas)
                    if(v > 0.2f && v < 0.9f && s > 0.1f) {
                        avg_eye_color[0] += blue;
                        avg_eye_color[1] += green;
                        avg_eye_color[2] += red;
                        non_red_pixels++;
                    }
                }
            }
        }

        // Calculate average surrounding eye color
        if(non_red_pixels > 0) {
            avg_eye_color[0] /= non_red_pixels;
            avg_eye_color[1] /= non_red_pixels;
            avg_eye_color[2] /= non_red_pixels;
        } else {
            // Fallback to a dark gray if no suitable pixels found
            avg_eye_color = Vec3f(60, 60, 60);
        }

        // Clean up red-eye mask with morphological operations
        int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        neighborhood_structure n8 = {di, dj, 8};

        // First close to connect regions
        red_mask = closing(red_mask, n8, 1);

        // Then open to remove small noise
        red_mask = opening(red_mask, n8, 1);

        // Find the center of the red-eye region for gradient correction
        int center_r = 0, center_c = 0;
        int total_red_pixels = 0;

        for(int r = 0; r < red_mask.rows; r++) {
            for(int c = 0; c < red_mask.cols; c++) {
                if(red_mask.at<uchar>(r, c) == 0) {
                    center_r += r;
                    center_c += c;
                    total_red_pixels++;
                }
            }
        }

        if(total_red_pixels > 0) {
            center_r /= total_red_pixels;
            center_c /= total_red_pixels;
        } else {
            // Fallback to center of region if no red pixels found
            center_r = red_mask.rows / 2;
            center_c = red_mask.cols / 2;
        }

        // Compute the maximum distance from center for gradient normalization
        float max_dist = 0;
        for(int r = 0; r < red_mask.rows; r++) {
            for(int c = 0; c < red_mask.cols; c++) {
                if(red_mask.at<uchar>(r, c) == 0) {
                    float dist = sqrt(pow(r - center_r, 2) + pow(c - center_c, 2));
                    max_dist = max(max_dist, dist);
                }
            }
        }

        // Apply correction to red-eye pixels with gradient and natural eye color
        for(int r = 0; r < red_mask.rows; r++) {
            for(int c = 0; c < red_mask.cols; c++) {
                if(red_mask.at<uchar>(r, c) == 0) {
                    int img_r = r + r_min;
                    int img_c = c + c_min;
                    Vec3b pixel = image.at<Vec3b>(img_r, img_c);

                    // Calculate distance from center for gradient effect
                    float dist = sqrt(pow(r - center_r, 2) + pow(c - center_c, 2));
                    float dist_ratio = max_dist > 0 ? dist / max_dist : 0;

                    // Gradient factor (darker in center, more original color at edges)
                    float gradient_factor = 0.5f + 0.5f * dist_ratio;

                    // Calculate luminance using proper weights
                    int luminance = (int)(pixel[0] * 7 + pixel[1] * 72 + pixel[2] * 21) / 100;

                    // Calculate corrected pixel value
                    // Blend between dark pupil color and average iris color
                    Vec3b corrected;

                    // Preserve very bright spots (specular highlights)
                    if(luminance > 200) {
                        corrected = pixel;
                    } else {
                        // Dark pupil color (based on average eye color but darker)
                        Vec3f dark_pupil = Vec3f(
                                avg_eye_color[0] * 0.5f,
                                avg_eye_color[1] * 0.5f,
                                avg_eye_color[2] * 0.5f
                        );

                        // Blend between dark pupil and original based on gradient
                        corrected[0] = min(255, (int)(dark_pupil[0] * (1.0f - gradient_factor) + pixel[0] * gradient_factor));
                        corrected[1] = min(255, (int)(dark_pupil[1] * (1.0f - gradient_factor) + pixel[1] * gradient_factor));
                        corrected[2] = min(255, (int)(dark_pupil[2] * (1.0f - gradient_factor) + pixel[2] * gradient_factor * 0.5f));
                    }

                    // Apply the correction to the result image
                    result.at<Vec3b>(img_r, img_c) = corrected;
                }
            }
        }

        // Apply edge-aware blending for smooth transitions
        int blend_radius = 1;
        for(int r = 0; r < red_mask.rows; r++) {
            for(int c = 0; c < red_mask.cols; c++) {
                if(red_mask.at<uchar>(r, c) == 255) {
                    // Check if it's near a red-eye pixel for blending
                    bool near_red_eye = false;
                    for(int br = -blend_radius; br <= blend_radius && !near_red_eye; br++) {
                        for(int bc = -blend_radius; bc <= blend_radius && !near_red_eye; bc++) {
                            int check_r = r + br;
                            int check_c = c + bc;
                            if(check_r >= 0 && check_r < red_mask.rows &&
                               check_c >= 0 && check_c < red_mask.cols &&
                               red_mask.at<uchar>(check_r, check_c) == 0) {
                                near_red_eye = true;
                            }
                        }
                    }

                    // Apply blending if it's near a red-eye pixel
                    if(near_red_eye) {
                        int img_r = r + r_min;
                        int img_c = c + c_min;
                        Vec3b original = image.at<Vec3b>(img_r, img_c);
                        Vec3b corrected = result.at<Vec3b>(img_r, img_c);

                        Vec3b blended;
                        blended[0] = (original[0] + corrected[0]) / 2;
                        blended[1] = (original[1] + corrected[1]) / 2;
                        blended[2] = (original[2] + corrected[2]) / 2;

                        result.at<Vec3b>(img_r, img_c) = blended;
                    }
                }
            }
        }
    }

    return result;
}

Mat draw_bounding_boxes(Mat image, vector<circumscribed_rectangle_coord> boxes, Scalar color) {
    Mat result = image.clone();

    for(size_t i = 0; i < boxes.size(); i++) {
        // Draw rectangle
        rectangle(result,
                  Point(boxes[i].c_min, boxes[i].r_min),
                  Point(boxes[i].c_max, boxes[i].r_max),
                  color, 2);
    }

    return result;
}

int main() {
    // Load image
    Mat image = imread("C:\\Users\\vladm\\CLionProjects\\RedEyeCorrection\\images\\far.jpg");
    if(image.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    imshow("Original Image", image);

    vector<circumscribed_rectangle_coord> faces = detect_faces(image);

    if(faces.empty()) {
        cout << "No faces detected" << endl;
        return 0;
    }

    Mat face_detection = draw_bounding_boxes(image, faces, Scalar(0, 255, 0));  // Green boxes for faces
    imshow("Face Detection", face_detection);

    vector<circumscribed_rectangle_coord> all_eyes;
    for(size_t i = 0; i < faces.size(); i++) {
        vector<circumscribed_rectangle_coord> eyes = detect_eyes(image, faces[i]);
        all_eyes.insert(all_eyes.end(), eyes.begin(), eyes.end());
    }

    Mat eye_detection = draw_bounding_boxes(face_detection, all_eyes, Scalar(255, 0, 0));  // Blue boxes for eyes
    imshow("Eye Detection", eye_detection);

    Mat corrected_image = detect_and_correct_red_eyes(image, all_eyes);

    imshow("Red Eye Correction", corrected_image);

    int width = image.cols;
    int height = image.rows;
    Mat comparison = Mat(height, width*2, image.type());

    image.copyTo(comparison(Rect(0, 0, width, height)));
    corrected_image.copyTo(comparison(Rect(width, 0, width, height)));

    imshow("Before and After", comparison);

    waitKey(0);
    return 0;
}