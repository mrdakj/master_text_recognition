#include <iostream>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <cassert>
#include <optional>
#include <limits>
#include <fdeep/fdeep.hpp>
#include "jamspell/spell_corrector.hpp"
#include "include/image.h"
#include "include/line_segmentation.h"

struct recognition_models {
    // collection of models used for letters recognition

    recognition_models()
        : model_one(fdeep::load_model("../model/one.json", true, fdeep::dev_null_logger))
        , model_one_two(fdeep::load_model("../model/one_two.json", true, fdeep::dev_null_logger))
        , model_first(fdeep::load_model("../model/first.json", true, fdeep::dev_null_logger))
        , model_second(fdeep::load_model("../model/second.json", true, fdeep::dev_null_logger))
    {
    }

    // model for single letter prediction
    // a, b, c, ..., z
    const fdeep::model model_one;
    // model for one vs two letters prediction
    // a, ..., z -> one letter
    // aa, ..., az, ..., za, ..., zz -> two letters
    const fdeep::model model_one_two;
    // model for the first bigram letter prediction
    // aa, ab, ..., az -> a
    const fdeep::model model_first;
    // model for the second bigram letter prediction
    // aa, ba, ..., za -> a
    const fdeep::model model_second;
};

struct Component {
    Component(image img, borders bounding_box, int dot_point, bool is_bigram = false)
        : img(std::move(img))
        , bounding_box(std::move(bounding_box))
        , dot_point(dot_point)
        , is_bigram(is_bigram)
        , one_letter(' ')
        , bigram({' ', ' '})
        , first(' ')
        , second(' ')
    {
    }

    image img;
    borders bounding_box;
    // dot point is the middle (horizontal) point of the upper half part of the component
    int dot_point;
    bool is_bigram;
    char one_letter;
    std::pair<char, char> bigram;
    char first;
    char second;
};

class image_recognizer {
public:
    image_recognizer(bool use_dictionary = true, bool debug = false)
        : m_dictionary(get_dictionary())
        , m_use_spell_check(use_dictionary)
        , m_debug(debug)
        , m_output_dir("out")
    {
        if (m_use_spell_check) {
            // load spell check model for english language
            m_corrector.LoadLangModel("../model/en.bin");
        }

        fs::create_directory(m_output_dir);
    }

    void recognize_image(const fs::path& img_name)
    {
        try {
            std::cout << img_name << std::endl;
            // set temporary vars
            set_tmp(img_name);

            if (m_debug) {
                fs::create_directory(m_debug_output_tmp);
            }

            image img(img_name);
            // segment lines
            auto result_image = segment_lines(img);

            if (m_debug) {
                result_image.save(m_debug_output_tmp / img_name.filename());
            }

            get_lines_text(result_image);
            // restore temporary vars
            clear_tmp();
        } 
        catch (std::exception& ex) {
            std::cerr << ex.what() << std::endl;
        }
    }

private:
    image segment_lines(const image& img) const
    {
        // returns image with line separating paths
        strips s(img);
        s.connect();
        auto strips_image = s.concatenate_strips();
        auto strips_contour = s.concatenate_strips_with_lines(false);
        auto strips_contour_original = s.concatenate_strips_with_lines(true);
        auto result_image = s.result();
        return result_image;
    }

    void set_tmp(const fs::path& img_name)
    {
        // set output vars for this image
        m_output_file_tmp = m_output_dir / (img_name.stem().string() + ((m_use_spell_check) ? "_dictionary.txt" : ".txt"));
        if (m_debug) {
            m_debug_output_tmp = m_output_dir / (img_name.stem().string() + "_debug");
        }
    }

    void clear_tmp()
    {
        // clear output vars
        m_output_file_tmp.clear();
        m_debug_output_tmp.clear();
    }

    std::unordered_set<std::string> get_dictionary() const
    {
        // read dictionary of english words and store it in a hash
        std::unordered_set<std::string> dictionary;
        std::ifstream infile("../dictionary/google-10000-english.txt");
        std::string word;

        while (std::getline(infile, word)) {
            dictionary.insert(word);
        }

        return dictionary;
    }

    std::pair<std::vector<pixel>, borders> bfs_get_component(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited) const
    {
        // get component pixels and bounding box
        std::queue<pixel> q;
        q.push(p);
        visited.insert(p);

        std::vector<pixel> black;

        // left, right, top, bottm
        borders b{p.i, p.i, p.j, p.j};

        while (!q.empty()) {
            auto current_pixel = q.front();
            black.push_back(current_pixel);
            q.pop();
            b.update(current_pixel);

            for (int dj : {-1, 0, 1}) {
                for (int di : {-1, 0, 1}) {
                    pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                    // if neighbour is black and not yet visited
                    if (img.check_color(neighbour, Color::black) && visited.find(neighbour) == visited.cend()) {
                        q.push(neighbour);
                        visited.insert(neighbour);
                    }
                }
            }
        }

        return {black, b};
    }

    bool is_dot(const Component& component, const std::vector<Component> & all_components) const
    {
        if (!(component.img.rows() < 20 && component.img.cols() < 20)) return false;

        int dot_middle = (component.bounding_box.left() + component.bounding_box.right()) / 2;
        int min_distance = std::numeric_limits<int>::max();
        int component_index = 0;

        // find the nearest component according to horizontal distance
        for (int component_i = 0; component_i < all_components.size(); ++component_i) {
            auto& other_pair = all_components[component_i];
            if (other_pair.bounding_box == component.bounding_box) {
                continue;
            }

            int distance = std::abs(dot_middle - other_pair.dot_point);
            if (distance < min_distance) {
                min_distance = distance;
                component_index = component_i;
            }
        }

        // add component (potential dot) to the top of the nearest component (potential part of letter i or j)
        auto glued_img = glue(all_components[component_index], component);
        recognize(glued_img);

        return  // merged component is i or j
                glued_img.one_letter == 'i' || 
                glued_img.one_letter == 'j' ||
                // merged component is bigram that contains i or j
                glued_img.bigram.first == 'i' || 
                glued_img.bigram.second == 'i' ||
                glued_img.bigram.first == 'j' ||
                glued_img.bigram.second == 'j';
    }

    std::pair<std::vector<Component>, std::vector<Component>> get_components_and_dots(const std::vector<Component> & all_components) const
    {
        // separate all components into dots and non-dot components
        std::vector<Component> components;
        std::vector<Component> dots;

        for (auto component : all_components) {
            if (is_dot(component, all_components)) {
                dots.emplace_back(std::move(component));
            }
            else {
                components.emplace_back(std::move(component));
            }
        }

        return {components, dots};
    }

    Component glue(Component component, Component dot) const
    {
        component.img.add_border(Direction::right, dot.bounding_box.right() - component.bounding_box.right());
        component.img.add_border(Direction::left, component.bounding_box.left() - dot.bounding_box.left());
        dot.img.add_border(Direction::right,  component.bounding_box.right() - dot.bounding_box.right());
        dot.img.add_border(Direction::left,  dot.bounding_box.left() - component.bounding_box.left());
        component.img.add_border(Direction::up, component.bounding_box.top() - dot.bounding_box.bottom());

        auto concatenated_img = component.img.concatenate_horizontal(dot.img);

        // update borders of the new component
        auto box = borders(
                std::min(component.bounding_box.left(), dot.bounding_box.left()),
                std::max(component.bounding_box.right(), dot.bounding_box.right()),
                dot.bounding_box.top(),
                component.bounding_box.bottom());

        return Component(std::move(concatenated_img), std::move(box), component.dot_point);
    }

    std::pair<double, double> get_avg_width_height(const std::vector<Component>& components) const
    {
        int height_sum = 0;
        int width_sum = 0;
        for (const auto& component : components) {
            height_sum += component.img.rows();
            width_sum += component.img.cols();
        }
        double height_avg = (double)height_sum/(double)components.size();
        double width_avg = (double)width_sum/(double)components.size();

        return {width_avg, height_avg};
    }

    std::pair<int, double> get_space_sum_and_avg(const std::vector<Component>& components, double width_avg) const
    {
        int space_sum = 0;
        for (int i = 0; i < (int)components.size()-1; ++i) {
            // space should be in interval [0, 3*average width]
            space_sum += std::min((int)(3*width_avg), 
                                   std::max(components[i+1].bounding_box.left() - components[i].bounding_box.right(), 0));
        }

        double space_avg = 0;
        if (components.size() > 1) {
            space_avg = (double)space_sum/(double)(components.size()-1);
        }

        return {space_sum, space_avg};
    }


    std::optional<std::string> select_word(const std::vector<std::string>& word_candidates) const
    {
        // choose a word that exists in the dictionary
        for (const auto& word : word_candidates) {
            if (m_dictionary.find(word) != m_dictionary.end()) {
                return std::make_optional(word);
            }
        }

        return std::nullopt;
    }

    void update_candidates_and_default_word(std::vector<std::string> & word_candidates, std::string & default_word, const Component & component, bool above_avg_height) const
    {
        // update word candidates and default word with new letter(s)
        default_word += component.one_letter;

        std::vector<std::string> words_candidates_to_add;
        for (auto& word : word_candidates) {
            std::string word_append = word;

            if (component.is_bigram) {
                word_append += component.bigram.first;
                word_append += component.bigram.second;
                words_candidates_to_add.emplace_back(word + std::string(1, component.one_letter));
            }
            else {
                word_append += component.one_letter;
            }

            if (component.first != ' ') {
                words_candidates_to_add.emplace_back(word + std::string(1, component.first));
            }

            if (component.second != ' ') {
                words_candidates_to_add.emplace_back(word + std::string(1, component.second));
            }

            if (component.one_letter == 'l' && !above_avg_height) {
                words_candidates_to_add.emplace_back(word + "c");
                words_candidates_to_add.emplace_back(word + "e");
            }
            if (component.one_letter == 'j' && !above_avg_height) {
                words_candidates_to_add.emplace_back(word + "i");
            }
            if (component.one_letter == 'p' && !above_avg_height) {
                words_candidates_to_add.emplace_back(word + "e");
            }

            word = std::move(word_append);
        }

        for (auto& word : words_candidates_to_add) {
            word_candidates.emplace_back(std::move(word));
        }
    }

    void merge_dots_and_components(std::vector<Component> & dots, std::vector<Component> & components) const
    {
        for (auto& dot : dots) {
            int dot_middle = (dot.bounding_box.left() + dot.bounding_box.right()) / 2;
            auto component = std::min_element(components.begin(), components.end(), [&](auto const& lhs, auto const& rhs){
                    return std::abs(dot_middle - lhs.dot_point) < std::abs(dot_middle - rhs.dot_point);
            });

            // add dot to the top of the component
            *component = glue(std::move(*component), std::move(dot));
        }
    }

    void sort_components(std::vector<Component> & components) const
    {
        std::sort(components.begin(), components.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.dot_point < rhs.dot_point;
        });
    }

    std::vector<Component> get_components(const image& img) const
    {
        std::vector<Component> all_components;
        std::unordered_set<pixel, pixel::hash> visited;

        for (int j = 0; j < img.rows(); ++j) {
            for (int i = 0; i < img.cols(); ++i) {
                pixel p{j,i};
                // if pixel is black and not yet visited
                if (img.check_color(p, Color::black) && visited.find(p) == visited.cend()) {
                    auto [pixels, b] = bfs_get_component(img, p, visited);
                    image im(b.height(), b.width());
                    for (auto pp : pixels) {
                        im(pp.j - b.top(), pp.i - b.left()) = img(pp);
                    }

                    // find minimum left and maximum right point of the upper half part of the component
                    int min_j = 0;
                    int max_j = 0;
                    for (int j_pom = 0; j_pom < im.cols(); ++j_pom) {
                        for (int i_half = 0; i_half < im.rows()/2; ++i_half) {
                            if (im.check_color({i_half, j_pom}, Color::black)) {
                                if (min_j == 0) {
                                    min_j = j_pom;
                                }
                                max_j = j_pom;
                            }
                        }
                    }

                    int dot_point = b.left() + (min_j + max_j)/2;
                    all_components.emplace_back(std::move(im), b, dot_point);

                }
            }
        }

        return all_components;
    }

    std::string get_line_text(const image& img, int line_num) const
    {
        if (m_debug) {
            fs::create_directory(m_debug_output_tmp / std::to_string(line_num));
        }

        auto all_components = get_components(img);

        auto [components, dots] = get_components_and_dots(all_components);
        merge_dots_and_components(dots, components);
        sort_components(components);

        auto [width_avg, height_avg] = get_avg_width_height(components);
        auto [space_sum, space_avg] = get_space_sum_and_avg(components, width_avg);

        auto is_word_end = [&](int i) {
            int this_diff = std::min((int)(3*width_avg), 
                                     std::max(components[i+1].bounding_box.left() - components[i].bounding_box.right(), 0));

            int prev_diff = (i > 0) ? std::min((int)(3*width_avg), 
                                                std::max(components[i].bounding_box.left() - components[i-1].bounding_box.right(), 0)) 
                                    : 1.5*space_avg;

            int next_diff = (i+2 < (int)components.size()) ?  std::min((int)(3*width_avg), 
                                                              std::max(components[i+2].bounding_box.left() - components[i+1].bounding_box.right(), 0)) 
                                                           : 1.5*space_avg;

            return (this_diff >= 1.5*prev_diff || this_diff >= 1.5*next_diff) && 
                   this_diff > 1.1*width_avg && 
                   this_diff >= 1.5*space_avg;
        };

        std::string line;
        int image_count = 0;
        std::string default_word;
        std::vector<std::string> words_candidates = {""};

        // word_ends[i] is true iff the ith component is the end of a word
        std::vector<bool> word_ends;

        for (int i = 0; i < (int)components.size(); ++i) {
            recognize(components[i]);
            word_ends.emplace_back(is_word_end(i));
        }

        for (int i = 0; i < (int)components.size() - 1; ++i) {
            if (!word_ends[i]) {
                auto merged = merge(components[i], components[i+1]);
                if (merged) {
                    recognize(*merged);
                    components[i].first = merged->bigram.first;
                    components[i+1].second = merged->bigram.second;
                    // merged->img.show();
                    // cv::waitKey(0);
                }
            }
        }

        for (int i = 0; i < (int)components.size(); ++i) {
            if (m_debug) {
                components[i].img.save(m_debug_output_tmp / std::to_string(line_num) / (std::to_string(image_count) + ".png"));
                ++image_count;
            }

            update_candidates_and_default_word(words_candidates, default_word, components[i], components[i].img.rows() > height_avg);

            if (i == (int)components.size()-1 || word_ends[i]) {
                auto selected_word = select_word(words_candidates);
                line += (selected_word) ? *selected_word : default_word;

                if (i != (int)components.size()-1) {
                    line += " ";
                    default_word = "";
                    words_candidates = {""};
                }
            }
        }

        return line;
    }

    std::optional<image> bfs_segment_line(const image& img, pixel p) const
    {
        // create new image that contains only one line using flood fill algorithm
        std::unordered_set<pixel, pixel::hash> visited;
        std::queue<pixel> q;
        q.push(p);
        visited.insert(p);

        int black_count = 0;

        // left, right, top, bottm
        borders b{p.i, p.i, p.j, p.j};

        while (!q.empty()) {
            auto current_pixel = q.front();
            q.pop();
            b.update(current_pixel);

            for (int dj : {-1, 0, 1}) {
                for (int di : {-1, 0, 1}) {
                    if (std::abs(dj) == std::abs(di)) {
                        continue;
                    }

                    pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                    if ((img.check_color(neighbour, Color::black) || img.check_color(neighbour, Color::white)) && 
                        visited.find(neighbour) == visited.cend()) {
                        if (img.check_color(neighbour, Color::black)) {
                            ++black_count;
                        }
                        q.push(neighbour);
                        visited.insert(neighbour);
                    }
                }
            }
        }

        // it is possible that we have only white pixels between separating line paths
        // we should skip that line
        if (black_count > 0) {
            image line(b.height(), b.width());
            for (auto pp : visited) {
                line(pp.j - b.top(), pp.i - b.left()) = img(pp);
            }
            line.crop();
            return std::make_optional(line);
        }

        return std::nullopt;
    }

    void get_lines_text(const image& img) const
    {
        std::ofstream output(m_output_file_tmp);

        int line_num = 0;
        for (int j = 1; j < img.rows(); ++j) {
            // gray color represents separating line path
            if (j == 1 || img.check_color({j-1,0}, Color::gray)) {
                auto maybe_line = bfs_segment_line(img, {j,0});

                // skip if line contains only white pixels
                if (maybe_line) {
                    if (m_debug) {
                        maybe_line->save(m_debug_output_tmp / ("line" + std::to_string(line_num) + ".png"));
                    }
                    auto line_text = get_line_text(*maybe_line, line_num);

                    if (m_use_spell_check) {
                        // pass the entire line text to spell checker
                        auto line_fixed = m_corrector.FixFragment(std::wstring(line_text.begin(), line_text.end()));
                        std::cout << std::string(line_fixed.begin(), line_fixed.end()) << std::endl;
                        output << std::string(line_fixed.begin(), line_fixed.end()) << std::endl;
                    }
                    else {
                        std::cout << line_text << std::endl;
                        output << line_text << std::endl;
                    }

                    ++line_num;
                }
            }
        }
    }

    void recognize(Component & component) const
    {
        image reized_img = component.img;
        reized_img.resize(28,28);
        auto t = reized_img.get_tensor(0,1);

        if (component.is_bigram) {
            component.bigram = {char(m_models.model_first.predict_class({t}) + 97), char(m_models.model_second.predict_class({t}) + 97)};
        }
        else {
            component.one_letter =  char(m_models.model_one.predict_class({t}) + 97);

            if (m_models.model_one_two.predict_class({t}) == 1) {
                component.is_bigram = true;
                component.bigram = {char(m_models.model_first.predict_class({t}) + 97), char(m_models.model_second.predict_class({t}) + 97)};
            }
        }
    }

    std::optional<Component> merge(const Component& img_a, const Component& img_b) const
    {
        // merge images a and b into ab
        char label_a = img_a.one_letter;
        char label_b = img_b.one_letter;
        int expected_number_of_components = 0;
        if (label_a != 'i' && label_a != 'j') {
            if (label_b != 'i' && label_b != 'j') {
                expected_number_of_components = 1;
            }
            else {
                expected_number_of_components = 2;
            }
        }
        else {
            if (label_b != 'i' && label_b != 'j') {
                expected_number_of_components = 2;
            }
            else {
                expected_number_of_components = 3;
            }
        }

        int step = 0;
        int offset_a = std::max(0, img_b.bounding_box.bottom() - img_a.bounding_box.bottom());
        int offset_b = std::max(0, img_a.bounding_box.bottom() - img_b.bounding_box.bottom());

        while (true) {
            auto result = merge(img_a.img, img_b.img, step, offset_a, offset_b);
            step += 1;
            if (step >= img_a.img.rows() || result.img.cols() - step < 0) {
                return std::nullopt;
            }

            if (number_of_components_expected(result.img, expected_number_of_components)) {
                result = merge(img_a.img, img_b.img, step+1, offset_a, offset_b);
                return std::optional<Component>(result);
            }
        }

        return std::nullopt;
    }

    Component merge(const image& img_a, const image& img_b, int step, int offset_a, int offset_b) const
    {
        image result(std::max(img_a.rows()+offset_a, img_b.rows()+offset_b), img_a.cols()+img_b.cols());

        int current_result_j = result.rows()-1;
        for (int j = img_a.rows()-1; j >= 0; --j) {
            for (int i = 0; i < img_a.cols(); ++i) {
                result(current_result_j-offset_a, i) = img_a(j,i);
            }
            --current_result_j;
        }

        current_result_j = result.rows()-1;
        for (int j = img_b.rows()-1; j >= 0 ; --j) {
            for (int i = 0; i < img_b.cols(); ++i) {
                pixel p{current_result_j-offset_b, i+img_a.cols()-step};
                if (result.check_color(p, Color::white)) {
                    result(p) = img_b(j,i);
                }
            }
            --current_result_j;
        }

        result.crop();
        return Component(std::move(result), {-1,-1,-1,-1}, -1, true);
    }

    bool number_of_components_expected(const image& img, int expected_number_of_components) const
    {
        // check if the number of components in the image is expected
        int result = 0;
        std::unordered_set<pixel, pixel::hash> visited;

        for (int j = 0; j < img.rows(); ++j) {
            for (int i = 0; i < img.cols(); ++i) {
                pixel p{j,i};
                // if pixel is black and not yet visited
                if (img.check_color(p, Color::black)) {
                    if (visited.find(p) == visited.cend()) {
                        ++result;
                        if (result > expected_number_of_components) {
                            return false;
                        }
                        bfs_update_visited(img, p, visited);
                    }
                }
            }
        }

        return result == expected_number_of_components;
    }

    void bfs_update_visited(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited) const
    {
        std::queue<pixel> q;
        q.push(p);
        visited.insert(p);

        // left, right, top, bottm
        borders b{p.i, p.i, p.j, p.j};

        while (!q.empty()) {
            auto current_pixel = q.front();
            q.pop();

            for (int dj : {-1, 0, 1}) {
                for (int di : {-1, 0, 1}) {
                    if (std::abs(dj) == std::abs(di)) {
                        continue;
                    }
                    pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                    // if neighbour is black and not yet visited
                    if (img.check_color(neighbour, Color::black) && visited.find(neighbour) == visited.cend()) {
                        q.push(neighbour);
                        visited.insert(neighbour);
                    }
                }
            }
        }
    }

private:
    // models for letters recognition
    recognition_models m_models;
    // spell checker model
    NJamSpell::TSpellCorrector m_corrector;
    // dictionary of english words
    std::unordered_set<std::string> m_dictionary;
    // indicates if spell checker should be used
    bool m_use_spell_check;
    // indicates if intermediate results should be saved
    bool m_debug;
    // output directory
    fs::path m_output_dir;

    // debug output directory for the passed image
    fs::path m_debug_output_tmp;
    // output file for the passed image
    fs::path m_output_file_tmp;
};


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "./keras2cpp path_to_img [dictionary] [debug]" << std::endl;
        return -1;
    }

    recognition_models models;


    if (fs::is_directory(argv[1])) {
        image_recognizer recognizer_dictionary;
        image_recognizer recognizer(false);

        for (auto it = fs::directory_iterator(argv[1]); it != fs::directory_iterator(); ++it) {
            if (fs::path(*it).extension() == ".png") {
                recognizer.recognize_image(*it);
                recognizer_dictionary.recognize_image(*it);
            }
        }
    }
    else {
        bool use_dictionary = false;
        bool debug = false;

        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "dictionary") {
                use_dictionary = true;
            }
            else if (std::string(argv[i]) == "debug") {
                debug = true;
            }
        }

        image_recognizer recognizer(use_dictionary, debug);
        recognizer.recognize_image(argv[1]);
    }

    return 0;
}
