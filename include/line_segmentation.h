#ifndef LINE_SEGMENTATION_H
#define LINE_SEGMENTATION_H 

#include "image.h"

namespace fs = std::filesystem;

class contour {
    // contour represents one path that separate lines in image
public:
    contour() = default;

    void clear();
    bool empty() const;

    pixel last_pixel() const;

    void add(pixel p);
    void add(contour other);
    void add_segment(pixel a, pixel b);
    // move all the pixels from the contour to the set
    void move_to(std::unordered_set<pixel, pixel::hash>& paths);

    // returns true if there exists pixel in the contour that is in the set
    bool intersects(const std::unordered_set<pixel, pixel::hash>& paths) const;

private:
    std::vector<pixel>::iterator begin();
    std::vector<pixel>::iterator end();

    std::vector<pixel> m_path;
};

class strip {
    // strip is created from multiple consecutive columns of the original image
    // each strip has black and white rectangles and lines
    // black rectangles represent text area in the original image
    // lines are used as a reference while creating paths that will separate lines
public:
    class line;
    class rectangle;

    // create a strip from the original image part [0, image height] X [m_left, m_right)
    strip(const image& original_image, int left, int right);

    bool operator==(const strip& other) const;
    bool operator!=(const strip& other) const;

    // image of black and white rectanges of width (m_right-m_left)
    const image& img() const;
    // if original, create image of black and white rectanges with strip lines
    // if not original, create image with strip lines using original image
    image image_with_lines(bool original) const;

    // create strip lines in white rectangles that will be used to form paths
    void create_lines();

    const std::vector<rectangle>& rectangles() const;
    int rectangles_count() const;
    int white_rectangles_count() const;
    int white_rectangles_with_lines_count() const;
    // average height of white rectanges in the strip
    int white_height_mean() const;
    // get rectange that contains the given row
    const rectangle& get_rectangle(int row) const;
    void remove_rectangle_from_image(const rectangle& r);
    void add_black_rectangle_to_image(const rectangle& r);
    // update rectange vector after deleting/adding black rectangles
    void update_rectangles();
    // get white rectangles that touch the given rectangle edge (from neighbour strip)
    std::vector<rectangle> white_rectangles_intersect(const rectangle& a) const;
    // returns true if exists black rectangle that touches the given rectangle edge (from neighbour strip)
    bool exists_black_rectangle(const rectangle& r) const;
    // returns true if exists black rectangle that covers the given rectangle edge (from neighbour strip)
    bool black_rectangle(const rectangle& r) const;
    // number of pixels that belong to black rectangle in a range [start, end)
    int black_pixels_count(int start, int end) const;

    // get index of the nearest line from the row that belongs to some rectangle
    int get_nearest_line(const std::vector<rectangle>& rectangles, int row) const;
    // get the nearest empty row from the pixel that belongs to some rectangle
    int get_nearest_empty_line(const std::vector<rectangle>& rectangles, pixel p, Direction dir) const;

    std::vector<line>& lines();
    const std::vector<line>& lines() const;
    // get line endings
    std::pair<pixel, pixel> line_pixels(const line& l) const;
    // get line row
    int row(int line_index) const;
    // set line to used
    void set_used(int line_index);
    // check if line is used
    bool used(int line_index) const;

    bool in_range(pixel p) const;
    int left() const;
    int right() const;
    int width() const;

    // returns true if given index represents the top rectange
    bool top_rectangle(size_t rectangle_index) const;
    // returns true if given index represents the bottom rectange
    bool bottom_rectangle(size_t rectangle_index) const;
    // get rectangle that is above the given rectangle
    const rectangle& above_rectangle(size_t rectangle_index) const;
    // get rectangle that is below the given rectangle
    const rectangle& below_rectangle(size_t rectangle_index) const;

private:
    // create m_image
    image generate_strip_image(const image& img);

    // create m_rectangles vector
    std::vector<rectangle> get_rectangles(const image& img) const;
    void fill_white_rectangles(image& result);
    // calculate m_white_height_mean
    int get_white_height_mean(const std::vector<rectangle>& rectangles) const;

private:
    // [m_left, m_right)
    int m_left, m_right;

    const image& m_original_image;
    // image with rectangles of width (m_right-m_left)
    // rectangles are generated from original image part [0, image height] X [m_left, m_right)
    image m_image;

    std::vector<rectangle> m_rectangles;
    // average height of white rectanges
    int m_white_height_mean;

    // lines in white rectangles
    std::vector<line> m_lines;

public:
    class line {
    public:
        line(int row);

        int row() const;

        bool used() const;
        void set_used();

    private:
        // row of the strip where line is placed
        // row X [m_left, m_right)
        int m_row;
        bool m_used;
    }; // line

    class rectangle {
    public:
        rectangle(int top, int bottom, Color color);

        int top() const;
        int bottom() const;
        int height() const;

        bool check_color(Color color) const;
        void fill(Color color);

        int line_index() const;
        void set_line_index(int line_index);

    private:
        // [m_top, m_botom)
        int m_top, m_botom;
        // either black or white
        Color m_color;
        // index of line in m_lines that belongs to the rectange, or
        // -1 if the line doesn't exist in the rectangle
        int m_line_index;
    }; // rectangle
}; // strip

class strips {
    // in order to find paths in the original image, we will create multiple strips
    // strips don't intersect, unoion of strips covers the entire image 
    // each strip is processed separately (in strip class) to create black and white rectangles,
    // and then all together (in this class), because some transformations depend on neighbour strips
public:
    strips(image img);

    // create image of rectangles
    image concatenate_strips() const;
    // if not original, then create image of rectangles and strip lines
    // if original, then create image with strip lines
    image concatenate_strips_with_lines(bool original) const;
    // image with lines separated using paths
    image result() const;

    // create paths
    void connect();

private:
    bool create_strips(int strip_width);

    // add and remove black rectangles from the strips
    void filter_strips();
    bool should_be_deleted(size_t strip_index, const strip::rectangle& r) const;
    void delete_black_rectangles();
    bool should_be_added(size_t strip_index, size_t rectangle_index) const;
    void add_black_rectangles();

    // returns true if it is the index of the first strip from the left
    bool left_strip(size_t strip_index) const;
    // returns true if it is the index of the last strip from the left
    bool right_strip(size_t strip_index) const;

    // get strip that is on the right side of given strip
    const strip& next_right_strip(size_t strip_index) const;
    // get strip that is on the left side of given strip
    const strip& next_left_strip(size_t strip_index) const;

    int strips_count() const;

    // strip that has the most strip lines
    strip& get_core_strip();
    // get a strip that contains the given column
    strip& get_strip(int col);

    // methods used for creating the candidate path
    void continue_path(pixel last_pixel, Direction dir);
    std::vector<strip::rectangle> get_rectangles_of_interest(
        const strip& last_strip,
        const strip::rectangle& last_rectangle,
        const strip& next_strip,
        const strip::rectangle& next_rectangle) const;
    pixel connect_with_existing_line(pixel last_pixel, Direction dir);
    pixel continue_with_strait_line(pixel last_pixel, Direction dir);
    pixel try_to_connect(
        const strip& last_strip,
        pixel last_pixel,
        const strip& next_strip,
        pixel next_pixel,
        int next_row,
        Direction dir);
    pixel go_around_and_cut(pixel last_pixel, Direction dir);
    pixel go_around_and_cut_helper(pixel current_pixel, Direction dir);
    bool bfs_go_around(pixel p, const std::unordered_set<pixel, pixel::hash>& candidates, Direction dir);
    borders bfs_get_box(pixel p) const;
    std::pair<borders, std::unordered_set<pixel, pixel::hash>> bfs_get_box_and_candidates(pixel p, Direction dir) const;

private:
    std::vector<strip> m_strips;
    // transformed original image (binarized, croped, etc.)
    image m_image;
    // average component height
    int m_component_height;
    // main paths we use as a result
    std::unordered_set<pixel, pixel::hash> m_paths;
    // current path we are creating
    // it can end up in main paths later
    contour m_candidate_path;
}; // strips

#endif /* LINE_SEGMENTATION_H */
