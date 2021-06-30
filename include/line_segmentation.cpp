#include "line_segmentation.h"

namespace fs = std::filesystem;

// contour
void contour::clear()
{
    m_path.clear();
}

bool contour::empty() const
{
    return m_path.empty();
}

pixel contour::last_pixel() const
{
    assert(!empty());
    return m_path.back();
}

void contour::add(pixel p)
{
    m_path.emplace_back(std::move(p));
}

void contour::add(contour other)
{
    m_path.insert(m_path.end(), std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
}

void contour::add_segment(pixel a, pixel b)
{
    // add segment [a, b] - it should be either horizontal or vertical segment
    assert(a.i == b.i || a.j == b.j);

    if (a.j == b.j) {
        for (int i = std::min(a.i, b.i); i <= std::max(a.i, b.i); ++i) {
            m_path.push_back({a.j, i});
        }
    }
    else if (a.i == b.i) {
        for (int j = std::min(a.j, b.j); j <= std::max(a.j, b.j); ++j) {
            m_path.push_back({j, a.i});
        }
    }
}

void contour::move_to(std::unordered_set<pixel, pixel::hash>& paths)
{
    for (auto& p : m_path) {
        paths.insert(std::move(p));
    }
    clear();
}

bool contour::intersects(const std::unordered_set<pixel, pixel::hash>& paths) const
{
    return std::any_of(m_path.begin(), m_path.end(), [&](const auto& p) { return paths.find(p) != paths.end(); });
}

std::vector<pixel>::iterator contour::begin()
{
    return m_path.begin();
}

std::vector<pixel>::iterator contour::end()
{
    return m_path.end();
}

// strip
// line
strip::line::line(int row)
    : m_row(row) 
    , m_used(false)
{}

int strip::line::row() const
{
    return m_row;
}

bool strip::line::used() const
{
    return m_used;
}

void strip::line::set_used()
{
    m_used = true;
}

// rectangle
strip::rectangle::rectangle(int top, int bottom, Color color)
    : m_top(top)
    , m_botom(bottom)
    , m_color(color)
    , m_line_index(-1)
{
}

int strip::rectangle::height() const
{
    return m_botom - m_top;
}

void strip::rectangle::fill(Color color)
{
    m_color = color;
}

bool strip::rectangle::check_color(Color color) const
{
    return m_color == color;
}

int strip::rectangle::top() const
{
    return m_top;
}

int strip::rectangle::bottom() const
{
    return m_botom;
}

int strip::rectangle::line_index() const
{
    return m_line_index;
}

void strip::rectangle::set_line_index(int line_index)
{
    m_line_index = line_index;
}

strip::strip(const image& original_image, int left, int right)
    : m_left(left)
    , m_right(right)
    , m_original_image(original_image)
    , m_image(generate_strip_image(m_original_image))
    , m_rectangles(get_rectangles(m_image))
    , m_white_height_mean(get_white_height_mean(m_rectangles))
{
}

int strip::white_rectangles_count() const
{
    return std::count_if(m_rectangles.begin(), m_rectangles.end(), [](const auto& r) { return r.check_color(Color::white); });
}

int strip::white_rectangles_with_lines_count() const
{
    return std::count_if(m_rectangles.begin(), m_rectangles.end(), [](const auto& r) { return r.line_index() != -1; });
}

std::vector<strip::rectangle> strip::get_rectangles(const image& img) const
{
    auto [height, width] = img.dim();

    std::vector<strip::rectangle> rectangles;
    int top = 0;
    Color color = static_cast<Color>(img(0,0));
    for (int j = 1; j < height; ++j) {
        if (!img.check_color({j, 0}, color)) {
            rectangles.push_back({top, j, color});
            color = (color == Color::white) ? Color::black : Color::white;
            top = j;
        }
    }
    rectangles.push_back({top, height, color});

    return rectangles;
}

int strip::get_white_height_mean(const std::vector<strip::rectangle>& rectangles) const
{
    auto height_sum = std::accumulate(rectangles.cbegin(), rectangles.cend(), 0, 
                      [&](int s, const auto& r) { return s + ((r.check_color(Color::white) && r.height() >= 3) ?  r.height() : 0); });
    auto white_rectangles_count = std::count_if(rectangles.cbegin(), rectangles.cend(),
                      [&](const auto& r) { return r.check_color(Color::white) && r.height() >= 3; });
    return (white_rectangles_count != 0) ?  height_sum / white_rectangles_count : 0;
}

const image& strip::img() const
{
    return m_image;
}

const std::vector<strip::rectangle>& strip::rectangles() const
{
    return m_rectangles;
}

bool strip::exists_black_rectangle(const strip::rectangle& r) const
{
    auto column_it = cvit::column_iterator(m_image);
    return std::any_of(cvit::row_iterator(*column_it, r.top()), cvit::row_iterator(*column_it, r.bottom()), [](const auto& x) { return x(0,0) == static_cast<unsigned char>(Color::black); });
}

bool strip::black_rectangle(const strip::rectangle& r) const
{
    auto column_it = cvit::column_iterator(m_image);
    return std::all_of(cvit::row_iterator(*column_it, r.top()), cvit::row_iterator(*column_it, r.bottom()), [](const auto& x) { return x(0,0) == static_cast<unsigned char>(Color::black); });
}

int strip::black_pixels_count(int start, int end) const
{
    auto column_it = cvit::column_iterator(m_image);
    return std::count_if(cvit::row_iterator(*column_it, start), cvit::row_iterator(*column_it, end), [](const auto& x) { return x(0,0) == static_cast<unsigned char>(Color::black); });
}

void strip::remove_rectangle_from_image(const strip::rectangle& r)
{
    m_image.fill(r.top(), r.bottom(), Color::white);
}

void strip::add_black_rectangle_to_image(const strip::rectangle& r)
{
    m_image.fill(r.top(), r.bottom(), Color::black);
}

void strip::update_rectangles()
{
    m_rectangles = get_rectangles(m_image);
    m_white_height_mean = get_white_height_mean(m_rectangles);
}

std::vector<strip::rectangle> strip::white_rectangles_intersect(const strip::rectangle& a) const
{
    std::vector<strip::rectangle> result;
    std::copy_if(m_rectangles.begin(), m_rectangles.end(), std::back_inserter(result), [&](const auto& r) {
        return r.check_color(Color::white) && !(r.bottom() <= a.top() || r.top() >= a.bottom());
    });
    return result;
}

int strip::get_nearest_line(const std::vector<strip::rectangle>& rectangles, int row) const
{
    auto result_it = std::min_element(rectangles.begin(), rectangles.end(), [&](const auto& lhs, const auto& rhs) { 
            int lhs_row = (lhs.line_index() != -1) ? m_lines[lhs.line_index()].row() : INT_MAX;
            int rhs_row = (rhs.line_index() != -1) ? m_lines[rhs.line_index()].row() : INT_MAX;
            return std::abs(lhs_row - row) < std::abs(rhs_row - row);
    });

    if (result_it != rectangles.end()) {
        return result_it->line_index();
    }

    return -1;
}

int strip::get_nearest_empty_line(const std::vector<strip::rectangle>& rectangles, pixel p, Direction dir) const
{
    int result = -1;
    for (const auto& r : rectangles) {
        for (int row = r.top(); row < r.bottom(); ++row) {
            if ((dir == Direction::right && m_original_image.row_empty(row, p.i, m_right)) || (dir == Direction::left && m_original_image.row_empty(row, m_left, p.i+1))) {
                if (result == -1 || std::abs(p.j - row) < std::abs(result - p.j)) {
                    result = row;
                }
            }
        }
    }

    return result;
}

void strip::create_lines()
{
    for (auto& r : m_rectangles) {
        if (r.top() != 0 && r.check_color(Color::white)) {
            int T1 = r.top() + std::min(r.height()/3, width()/3);
            if (m_original_image.row_empty(T1, m_left, m_right)) {
                r.set_line_index(m_lines.size());
                m_lines.push_back({T1});
            }
            else {
                for (int dj = 1; T1 - dj >= r.top() || T1 + dj < r.bottom(); ++dj) {
                    if (T1 - dj >= r.top() && m_original_image.row_empty(T1-dj, m_left, m_right)) {
                        r.set_line_index(m_lines.size());
                        m_lines.push_back({T1-dj});
                        break;
                    }

                    if (T1 + dj < r.bottom() && m_original_image.row_empty(T1+dj, m_left, m_right)) {
                        r.set_line_index(m_lines.size());
                        m_lines.push_back({T1+dj});
                        break;
                    }
                }
            }
        }
    }
}

image strip::image_with_lines(bool original) const
{
    image result = (original) ? image::copy(m_original_image, m_left, m_right) : image::copy(m_image);

    for (const auto& line : m_lines) {
        result.fill_row(line.row(), Color::gray);
    }

    return result;
}

void strip::set_used(int line_index)
{
    m_lines[line_index].set_used();
}

bool strip::used(int line_index) const
{
    return m_lines[line_index].used();
}

int strip::row(int line_index) const
{
    return m_lines[line_index].row();
}

std::pair<pixel, pixel> strip::line_pixels(const line& l) const
{
    return {{l.row(), left()}, {l.row(), right()-1}};
}

std::vector<strip::line>& strip::lines()
{
    return m_lines;
}

const std::vector<strip::line>& strip::lines() const
{
    return m_lines;
}

const strip::rectangle& strip::get_rectangle(int row) const
{
    assert(row >= 0 && row < m_original_image.rows() && rectangles_count() > 0);
    auto it = std::find_if(m_rectangles.begin(), m_rectangles.end(), [&](const auto& r) { return row >= r.top() && row < r.bottom(); });
    assert(it != m_rectangles.end());
    return *it;
}

bool strip::operator==(const strip& other) const
{
    return m_left == other.left() && m_right == other.right();
}

bool strip::operator!=(const strip& other) const
{
    return !(*this == other);
}

bool strip::in_range(pixel p) const
{
    return p.i >= left() && p.i < right() && p.j >= 0 && p.j < m_image.rows();
}


int strip::left() const
{
    return m_left;
}

int strip::right() const
{
    return m_right;
}

int strip::white_height_mean() const
{
    return m_white_height_mean;
}

int strip::width() const
{
    return m_right - m_left;
}

bool strip::top_rectangle(size_t rectangle_index) const
{
    return rectangle_index == 0;
}

bool strip::bottom_rectangle(size_t rectangle_index) const
{
    return rectangle_index == m_rectangles.size()-1;
}

const strip::rectangle& strip::above_rectangle(size_t rectangle_index) const
{
    assert(!top_rectangle(rectangle_index));
    return m_rectangles[rectangle_index-1];
}

const strip::rectangle& strip::below_rectangle(size_t rectangle_index) const
{
    assert(!bottom_rectangle(rectangle_index));
    return m_rectangles[rectangle_index+1];
}

int strip::rectangles_count() const
{
    return m_rectangles.size();
}

image strip::generate_strip_image(const image& img)
{
    int strip_width = width();
    image result(img.rows(), strip_width);

    int j = 0;
    std::for_each(cvit::row_iterator(img), cvit::row_iterator(), [&](const auto& row) {
        int avg = row.sum(m_left, m_right) / strip_width;
        result.fill_row(j++, avg);
    });

    // result.show();
    // cv::waitKey(0);
    result.threshold(true);
    // result.show();
    // cv::waitKey(0);
    fill_white_rectangles(result);

    return result;
}

void strip::fill_white_rectangles(image& result)
{
    auto rectangles = get_rectangles(result);
    int height_threshold = get_white_height_mean(rectangles)/2;

    // result.show();
    // cv::waitKey(0);

    for (const auto& r : rectangles) {
        if (r.check_color(Color::white) && r.bottom() != m_image.rows() && r.height() <= height_threshold) {
            // fill the white rectangle which heigh is below a threshold and it is not the last one
            result.fill(r.top(), r.bottom(), Color::black);
        }
    }
    // result.show();
    // cv::waitKey(0);
}

// strips
strips::strips(image img)
    : m_image(std::move(img))
{
    // convert image to black and white
    m_image.threshold();

    // get average component size and remove noise
    auto [avg_component_width, avg_component_height] = m_image.get_component_avg_size_and_remove_noise();
    m_component_height = avg_component_height;

    // crop the image after noise is removed
    m_image.crop();
    // add padding of one
    m_image.add_border();

    // try to find the right strip width for the image
    // if strip width is too big, we will not have enough lines to form a correct path
    // if strip width is too small, we will have wrong lines that will create wrong paths
    // it is important to start from the bigger width
    for (int strip_width_factor = 8; strip_width_factor >= 1; --strip_width_factor) {
        int strip_width = strip_width_factor*avg_component_width;
        if (create_strips(strip_width)) {
            // we can stop since we succeeded to create a line in more than 50% white rectanges
            break;
        }
    }
}

bool strips::create_strips(int strip_width)
{
    int width = m_image.cols();
    m_strips.clear();
    // calculate strips width
    int strips_count = std::ceil(width/static_cast<double>(strip_width));
    int strip_width_min = std::floor(width/static_cast<double>(strips_count));
    int strip_width_max = std::ceil(width/static_cast<double>(strips_count));
    int strips_count_max = width - strips_count*strip_width_min;
    int strips_count_min = strips_count - strips_count_max;

    // create strips
    for (int i = 0; i < strips_count_max; ++i) {
        m_strips.emplace_back(m_image, i*strip_width_max, (i+1)*strip_width_max);
    }

    int offset = strips_count_max*strip_width_max;
    for (int i = 0; i < strips_count_min; ++i) {
        m_strips.emplace_back(m_image, offset + i*strip_width_min, offset + (i+1)*strip_width_min);
    }

    assert(m_strips.back().right() == width);
    if (strips_count_max != 0) {
        int max_count = std::count_if(m_strips.begin(), m_strips.end(), [&](const auto& s) { return s.width() == strip_width_max; });
        int min_count = std::count_if(m_strips.begin(), m_strips.end(), [&](const auto& s) { return s.width() == strip_width_min; });
        assert(max_count == strips_count_max);
        assert(min_count == strips_count_min);
        assert(max_count + min_count == static_cast<int>(m_strips.size()));
    }
    else 
    {
        int min_count = std::count_if(m_strips.begin(), m_strips.end(), [&](const auto& s) { return s.width() == strip_width_min; });
        assert(min_count == static_cast<int>(m_strips.size()));
    }

    // add/remove black rectangles from strips
    filter_strips();

    int white_rectangles_count = 0;
    int white_rectangles_with_lines_count = 0;
    for (auto& s : m_strips) {
        s.create_lines();
        white_rectangles_count += s.white_rectangles_count();
        white_rectangles_with_lines_count += s.white_rectangles_with_lines_count();
    }

    return white_rectangles_count > 0 && white_rectangles_with_lines_count/static_cast<double>(white_rectangles_count) >= 0.5;
}

image strips::concatenate_strips() const
{
    return std::accumulate(m_strips.begin(), m_strips.end(), image(m_image.rows(), 0), 
            [](image r, const auto& s) { return r + s.img(); });
}

image strips::concatenate_strips_with_lines(bool original) const
{
    return std::accumulate(m_strips.begin(), m_strips.end(), image(m_image.rows(), 0), 
            [&](image r, const auto& s) { return r + s.image_with_lines(original); });
}

strip& strips::get_core_strip()
{
    assert(!m_strips.empty());
    auto it = std::max_element(m_strips.begin(), m_strips.end(), 
            [](const auto& a, const auto& b) { return a.lines().size() < b.lines().size(); });
    assert(it != m_strips.end());
    return *it;
}

void strips::connect()
{
    // create paths from lines of the core strip
    auto& core_strip = get_core_strip();

    for (auto& l : core_strip.lines()) {
        // mark the line as used so we don't try to form a path from it later again
        l.set_used();

        auto [left_pixel, right_pixel] = core_strip.line_pixels(l);
        // clear a candidate path so we don't have stale data
        m_candidate_path.clear();
        m_candidate_path.add_segment(left_pixel, right_pixel);
        continue_path(right_pixel, Direction::right);
        continue_path(left_pixel, Direction::left);

        // add the candidate path to main paths
        m_candidate_path.move_to(m_paths);
    }

    // try to create paths from unused lines
    for (auto& s : m_strips) {
        for (auto& l : s.lines()) {
            if (l.used()) {
                continue;
            }
            l.set_used();

            auto [left_pixel, right_pixel] = s.line_pixels(l);
            // clear a candidate path so we don't have stale data
            m_candidate_path.clear();
            m_candidate_path.add_segment(left_pixel, right_pixel);
            continue_path(right_pixel, Direction::right);
            continue_path(left_pixel, Direction::left);

            if (!m_candidate_path.intersects(m_paths)) {
                // there is no intersection between a candidate path and main paths
                // add the candidate path to main paths
                m_candidate_path.move_to(m_paths);
            }
        }
    }
}

void strips::continue_path(pixel last_pixel, Direction dir)
{
    assert(m_image.in_range(last_pixel));

    // try to connect the last pixel with an existing unused line
    pixel new_last_pixel = connect_with_existing_line(last_pixel, dir);

    if (new_last_pixel == last_pixel) {
        // if we cannot connect with an existing line, try to create a new line
        // and connect the last pixel with it
        new_last_pixel = continue_with_strait_line(last_pixel, dir);
    }

    if (new_last_pixel == last_pixel) {
        // if we cannot continue the path with strait line,
        // try to find a path that goes around the obstacle or cut it
        new_last_pixel = go_around_and_cut(last_pixel, dir);
    }

    if (new_last_pixel != last_pixel) {
        // if we made a progress, continue
        continue_path(new_last_pixel, dir);
    }
}

std::vector<strip::rectangle> strips::get_rectangles_of_interest(
    const strip& last_strip,
    const strip::rectangle& last_rectangle,
    const strip& next_strip,
    const strip::rectangle& next_rectangle) const
{
    // last rectangle belongs to the last strip
    // next rectangle belongs to the next strip
    
    if (last_strip != next_strip) {
        // if the last and the next pixel are in different strips
        if (last_rectangle.check_color(Color::white)) {
            // if the last pixel was in a white rectangle, 
            // we want to go in a white rectangle in the next strip
            // that is touching with the last rectangle in the last strip
            return next_strip.white_rectangles_intersect(last_rectangle);
        }
        else {
            // if the last pixel was in a black rectangle, 
            // take the next rectangle if it is white,
            // otherwise we don't have rectangle of interest
            return (next_rectangle.check_color(Color::white)) ?  std::vector{next_rectangle} : std::vector<strip::rectangle>{};
        }
    }

    // if the last and the next pixel are in the same strip
    // both the last the next rectangle are the same
    
    // take the last (next) rectangle if it is white,
    // otherwise we don't have rectangle of interest
    return (last_rectangle.check_color(Color::white)) ?  std::vector{last_rectangle} : std::vector<strip::rectangle>{};
}

pixel strips::connect_with_existing_line(pixel last_pixel, Direction dir)
{
    // get next left or next right pixel
    auto next_pixel = last_pixel.next_pixel(dir);

    if (!m_image.in_range(next_pixel)) {
        return last_pixel;
    }

    const auto& last_strip = get_strip(last_pixel.i);
    const auto& last_rectangle = last_strip.get_rectangle(last_pixel.j);
    auto& next_strip = get_strip(next_pixel.i);
    const auto& next_rectangle = next_strip.get_rectangle(next_pixel.j);

    auto rectangles_of_interest = get_rectangles_of_interest(last_strip, last_rectangle, next_strip, next_rectangle);
    int candidate_line_index = next_strip.get_nearest_line(rectangles_of_interest, last_pixel.j);

    if (candidate_line_index != -1) {
        if (next_strip.used(candidate_line_index)) {
            // if the line is used don't try to go that way
            return last_pixel;
        }
        // mark it as used even if we don't connect it here
        // because we don't want to create path from it later
        // as it is probably not the real candidate
        next_strip.set_used(candidate_line_index);

        int line_row = next_strip.row(candidate_line_index);
        assert(line_row != -1);

        return try_to_connect(last_strip, last_pixel, next_strip, next_pixel, line_row, dir);
    }

    // we didn't find any line
    return last_pixel;
}

pixel strips::continue_with_strait_line(pixel last_pixel, Direction dir)
{
    // get next left or next right pixel
    auto next_pixel = last_pixel.next_pixel(dir);

    if (!m_image.in_range(next_pixel)) {
        return last_pixel;
    }

    const auto& last_strip = get_strip(last_pixel.i);
    const auto& last_rectangle = last_strip.get_rectangle(last_pixel.j);
    const auto& next_strip = get_strip(next_pixel.i);
    const auto& next_rectangle = next_strip.get_rectangle(next_pixel.j);

    auto rectangles_of_interest = get_rectangles_of_interest(last_strip, last_rectangle, next_strip, next_rectangle);
    int next_row = next_strip.get_nearest_empty_line(rectangles_of_interest, next_pixel, dir);

    return (next_row != -1) ? try_to_connect(last_strip, last_pixel, next_strip, next_pixel, next_row, dir) : last_pixel;
}

pixel strips::try_to_connect(
    const strip& last_strip,
    pixel last_pixel,
    const strip& next_strip,
    pixel next_pixel,
    int next_row,
    Direction dir)
{
    if (std::abs(next_row - next_pixel.j) > std::min(last_strip.white_height_mean(), m_component_height)) {
        // don't connect if it is too far from the last pixel
        return last_pixel;
    }

    if (!m_image.column_empty(next_pixel.i, std::min(next_pixel.j, next_row), std::max(next_pixel.j, next_row)+1)) {
        // don't connect if we will cross some black pixels in the original image
        return last_pixel;
    }

    // vertically connect the next pixel with the beginning of the next line
    m_candidate_path.add_segment(next_pixel, {next_row, next_pixel.i});

    // add the next line in the path
    if (dir == Direction::right) {
        m_candidate_path.add_segment({next_row, next_pixel.i}, {next_row, next_strip.right()-1});
        return pixel{next_row, next_strip.right()-1};
    }
    else {
        m_candidate_path.add_segment({next_row, next_strip.left()}, {next_row, next_pixel.i});
        return pixel{next_row, next_strip.left()};
    }

    return last_pixel;
}

pixel strips::go_around_and_cut(pixel last_pixel, Direction dir)
{
    // get next left or next right pixel
    auto next_pixel = last_pixel.next_pixel(dir);

    if (!m_image.in_range(next_pixel)) {
        return last_pixel;
    }

    const auto& next_strip = get_strip(next_pixel.i);

    while (next_strip.in_range(next_pixel)) {
        if (m_image.check_color(next_pixel, Color::white)) {
            // if it is a white pixel add it to the candidate path
            m_candidate_path.add(next_pixel);
        } 
        else {
            // it is is a black pixel try to find a path around the obstacle or cut it
            next_pixel = go_around_and_cut_helper(next_pixel, dir);
        }

        next_pixel = next_pixel.next_pixel(dir);
    }

    return m_candidate_path.last_pixel();
}

pixel strips::go_around_and_cut_helper(pixel current_pixel, Direction dir)
{
    // current pixel is black and it is not in a candidate path
    auto box = bfs_get_box(current_pixel);
    
    // find bounding box and candidate pixels for path if we go down
    auto [border_down, candidates_down] = bfs_get_box_and_candidates(current_pixel, Direction::down);
    int height_down = border_down.height();

    // find bounding box and candidate pixels for path if we go up
    auto [border_up, candidates_up] = bfs_get_box_and_candidates(current_pixel, Direction::up);
    int height_up = border_up.height();

    const auto& current_strip = get_strip(current_pixel.i);
    int d = std::min(current_strip.white_height_mean(), m_component_height);

    // number of pixels we will reach that belong to black rectangle in a current strip if we go down
    int black_pixels_in_strip_down = current_strip.black_pixels_count(border_down.top(), border_down.bottom());
    // number of pixels we will reach that belong to black rectangle in a current strip if we go up
    int black_pixels_in_strip_up = current_strip.black_pixels_count(border_up.top(), border_up.bottom());

    auto try_to_find_path = [&](Direction up_down, bool relaxed = false) {
        // this is some heuristics we use to decide if we should try to go down
        if (relaxed || (up_down == Direction::down && height_down <= d && (height_down <= d/2 || black_pixels_in_strip_down <= 0.25*height_down))) {
            // return true if path is found
            return bfs_go_around(current_pixel.previous_pixel(dir), candidates_down, dir);
        }

        // this is some heuristics we use to decide if we should try to go up
        if (relaxed || (up_down == Direction::up && height_up <= d && (height_up <= d/2 || black_pixels_in_strip_up <= 0.25*height_up))) {
            // return true if path is found
            return bfs_go_around(current_pixel.previous_pixel(dir), candidates_up, dir);
        }

        return false;
    };

    bool path_found = (box.bottom() - current_pixel.j <= current_pixel.j - box.top()) ? 
                        // first try to find path going down and then going up
                        (try_to_find_path(Direction::down) || try_to_find_path(Direction::up)) :
                        // first try to find path going up and then going down
                        (try_to_find_path(Direction::up) || try_to_find_path(Direction::down));

    if (!path_found) {
        path_found = (box.bottom() - current_pixel.j <= current_pixel.j - box.top()) ? 
                        // first try to find path going down and then going up
                        (try_to_find_path(Direction::down, true) || try_to_find_path(Direction::up, true)) :
                        // first try to find path going up and then going down
                        (try_to_find_path(Direction::up, true) || try_to_find_path(Direction::down, true));
    }

    if (!path_found) {
        // we need to cut
        // add all continuous black pixels in the candidate path
        while (m_image.check_color(current_pixel, Color::black)) {
            m_candidate_path.add(current_pixel);
            current_pixel = current_pixel.next_pixel(dir);
        }

        if (m_image.in_range(current_pixel)) {
            // add white pixel in the path so we don't examine this component again
            m_candidate_path.add(current_pixel);
        }
    }

    return m_candidate_path.last_pixel();
}

bool strips::bfs_go_around(pixel p, const std::unordered_set<pixel, pixel::hash>& candidates, Direction dir)
{
    // candidates are white pixels that are around component we hit, either below hitting point or above it
    
    // the goal is to find a path that starts in the pixel (p) just before the hitting point and
    // ends on the other side of the component in the same level as the hitting point
    // we can use only candidate pixels to form a path
    assert(dir == Direction::right || dir == Direction::left);

    std::unordered_set<pixel, pixel::hash> visited;
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    contour path;
    bool valid_path = false;

    auto process_neighbour = [&](pixel current_pixel, int dj, int di) {
        pixel neighbour{current_pixel.j + dj, current_pixel.i + di};
        
        if (candidates.find(neighbour) != candidates.cend()) {
            if (visited.find(neighbour) == visited.cend()) {
                // if neighbour is a candidate and not yet visited
                q.push(neighbour);
                visited.insert(std::move(neighbour));
            }
        }
    };

    auto stop_condition = [&](pixel current_pixel) {
        return // we reach the other side of the component at the same level of the hitting point
               (dir == Direction::right && current_pixel.j == p.j && current_pixel.i > p.i) ||
               (dir == Direction::left && current_pixel.j == p.j && current_pixel.i < p.i);
    };

    while (!q.empty()) {
        auto current_pixel = q.front();
        q.pop();
        path.add(current_pixel);

        if (stop_condition(current_pixel)) {
            valid_path = true;
            break;
        }

        process_neighbour(current_pixel, -1, 0);
        process_neighbour(current_pixel, 0, -1);
        process_neighbour(current_pixel, 0, 1);
        process_neighbour(current_pixel, 1, 0);
    }

    if (path.empty() || !valid_path) {
        return false;
    }

    m_candidate_path.add(std::move(path));
    return true;
}

borders strips::bfs_get_box(pixel p) const
{
    // get bounding box of component (part of it) we hit
    std::unordered_set<pixel, pixel::hash> visited;
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};

                if (m_image.check_color(neighbour, Color::black)) {
                    if (visited.find(neighbour) == visited.cend()) {
                        // if neighbour is black and not yet visited
                        q.push(neighbour);
                        visited.insert(std::move(neighbour));
                    }
                }
            }
        }
    }

    return b;
}

std::pair<borders, std::unordered_set<pixel, pixel::hash>> strips::bfs_get_box_and_candidates(pixel p, Direction dir) const
{
    // get bounding box of component (part of it) we hit, but only below or above the hitting point
    // if direction is down (up), top (bottom) border of the bounding box will be a row of the hitting point
    // get candidate pixels for potential path
    // candidate pixels are white and they are around component 
    assert(dir == Direction::down || dir == Direction::up);

    std::unordered_set<pixel, pixel::hash> candidates;
    std::unordered_set<pixel, pixel::hash> visited;
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};

                if (dir == Direction::down && neighbour.j < p.j) {
                    // don't go above the hitting row
                    continue;
                }

                if (dir == Direction::up && neighbour.j > p.j) {
                    // don't go below the hitting row
                    continue;
                }

                if (m_image.check_color(neighbour, Color::black)) {
                    if (visited.find(neighbour) == visited.cend()) {
                        // if neighbour is black and not yet visited
                        q.push(neighbour);
                        visited.insert(std::move(neighbour));
                    }
                }
                else if (m_image.check_color(neighbour, Color::white)) {
                    candidates.insert(std::move(neighbour));
                }
            }
        }
    }

    return {std::move(b), std::move(candidates)};
}

strip& strips::get_strip(int col)
{
    assert(col >= 0 && col < m_image.cols() && strips_count() > 0);
    auto it = std::find_if(m_strips.begin(), m_strips.end(), [&](const auto& s) { return col >= s.left() && col < s.right(); });
    assert(it != m_strips.end());
    return *it;
}

image strips::result() const
{
    image result_img = image::copy(m_image);

    for (const auto& p : m_paths) {
        result_img(p) = static_cast<unsigned char>(Color::gray);
    }

    return result_img;
}

bool strips::should_be_deleted(size_t strip_index, const strip::rectangle& r) const
{
    return (r.check_color(Color::black) && r.height() < 5) && 
           // there is no black rectangle on the left
           (strip_index == 0 || !m_strips[strip_index-1].exists_black_rectangle(r)) &&
           // there is no black rectangle on the right
           (strip_index == m_strips.size()-1 || !m_strips[strip_index+1].exists_black_rectangle(r));
}

void strips::delete_black_rectangles()
{
    // delete small black dangling rectangles
    for (int s = 0; s < strips_count(); ++s) {
        bool update_strip = false;
        for (const auto& r : m_strips[s].rectangles()) {
            if (should_be_deleted(s, r)) {
                m_strips[s].remove_rectangle_from_image(r);
                update_strip = true;
            }
        }

        if (update_strip) {
            m_strips[s].update_rectangles();
        }
    }
}

bool strips::should_be_added(size_t strip_index, size_t rectangle_index) const
{
    const auto& strip = m_strips[strip_index];
    const auto& r = strip.rectangles()[rectangle_index];
    return r.check_color(Color::white) && 
           // it is the first or the last rectangle in the strip, or there is some small black rectangle above or below it
           (strip.top_rectangle(rectangle_index) || strip.above_rectangle(rectangle_index).height() < 5 || 
            strip.bottom_rectangle(rectangle_index) || strip.below_rectangle(rectangle_index).height() < 5) &&
           // it is the first or the last strip, or there is a black rectange on the left or on the right
           (left_strip(strip_index) || next_left_strip(strip_index).black_rectangle(r)) &&
           (right_strip(strip_index) || next_right_strip(strip_index).black_rectangle(r));
}

void strips::add_black_rectangles()
{
    for (int s = 0; s < strips_count(); ++s) {
        bool update_strip = false;
        for (int rectangle_index = 0; rectangle_index < m_strips[s].rectangles_count(); ++rectangle_index) {
            if (should_be_added(s, rectangle_index)) {
                m_strips[s].add_black_rectangle_to_image(m_strips[s].rectangles()[rectangle_index]);
                update_strip = true;
            }
        }

        if (update_strip) {
            m_strips[s].update_rectangles();
        }
    }
}

void strips::filter_strips()
{
    delete_black_rectangles();
    add_black_rectangles();
}

bool strips::left_strip(size_t strip_index) const
{
    return strip_index == 0;
}

bool strips::right_strip(size_t strip_index) const
{
    return strip_index == m_strips.size()-1;
}

const strip& strips::next_right_strip(size_t strip_index) const
{
    assert(!right_strip(strip_index));
    return m_strips[strip_index+1];
}

const strip& strips::next_left_strip(size_t strip_index) const
{
    assert(!left_strip(strip_index));
    return m_strips[strip_index-1];
}

int strips::strips_count() const
{
    return m_strips.size();
}
